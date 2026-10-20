# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import re
import warnings
from collections import defaultdict
from enum import Enum

from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization.quant_args import DynamicType, QuantizationArgs
from compressed_tensors.quantization.quant_scheme import (
    QuantizationScheme,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.utils import is_module_quantized
from compressed_tensors.utils import find_unique_name
from compressed_tensors.utils.match import match_name, match_targets
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from torch.nn import Module


__all__ = [
    "QuantizationStatus",
    "QuantizationConfig",
    "LIFECYCLE_ORDER",
    "DEFAULT_QUANTIZATION_METHOD",
    "DEFAULT_QUANTIZATION_FORMAT",
]


def _map_to_checkpoint_names(model: Module, names: list[str]) -> list[str]:
    """Translate HF module names to checkpoint names for targets and ignores.

    Transformers v5 may rename weight keys on load (e.g. vision_embedder ->
    embed_vision).  Module names come from ``model.named_modules()``
    which uses HF names, but safetensors keys use checkpoint names.  This
    applies the same reverse mapping that ``save_pretrained`` uses for weights.
    """
    weight_conversions = getattr(model, "_weight_conversions", None)
    if not weight_conversions:
        return names

    inverted = [conv.reverse_transform() for conv in reversed(weight_conversions)]

    result = []
    for name in names:
        # Some conversions match a complete weight key or a trailing dot.
        # This assumes Linear-style `.weight` parameters, which is naive and not great.
        # Conversions specific to other parameter names are not handled.
        name += ".weight"
        for rev in inverted:
            renamed, matched = rev.rename_source_key(name)
            if matched is not None:
                name = renamed
        result.append(name.removesuffix(".weight"))

    return result


def _compact_checkpoint_targets(names: list[str]) -> list[str]:
    """Factor exact paths into regexes, preserving namespace prefixes."""
    if len(names) <= 1:
        return names

    trees = {}
    for name in sorted(set(names)):
        parts = name.split(".")
        # Keep the prefix before indexed layers literal for serving-side mappers.
        split = next(
            (i for i, part in enumerate(parts) if part.isdecimal()), len(parts) - 1
        )
        prefix = ".".join(parts[:split])
        node = trees.setdefault(prefix, {})
        for part in parts[split:]:
            node = node.setdefault(part, {})
        node[None] = {}

    def pattern(node: dict, separator: str = r"\.", digits: bool = False) -> str:
        # Only combine siblings with identical suffixes: independent unions of
        # each path component would introduce names that were never assigned.
        suffixes = defaultdict(list)
        for part, child in node.items():
            if part is not None:
                suffixes[pattern(child, "" if digits else r"\.", digits)].append(
                    re.escape(part)
                )
        branches = [""] if None in node else []
        for suffix, parts in suffixes.items():
            if len(parts) > 1 and all(
                part.isascii() and part.isdecimal() for part in parts
            ):
                if digits:
                    chars = "".join(sorted(parts))
                    if len(chars) > 2 and ord(chars[-1]) - ord(chars[0]) + 1 == len(
                        chars
                    ):
                        chars = f"{chars[0]}-{chars[-1]}"
                    parts = [f"[{chars}]"]
                else:
                    # Separate widths so compaction cannot introduce leading zeros.
                    digit_trees = {}
                    for number in parts:
                        child = digit_trees.setdefault(len(number), {})
                        for digit in number:
                            child = child.setdefault(digit, {})
                        child[None] = {}
                    parts = [pattern(tree, "", True) for tree in digit_trees.values()]
            prefix = parts[0] if len(parts) == 1 else f"(?:{'|'.join(parts)})"
            branches.append(separator + prefix + suffix)
        return branches[0] if len(branches) == 1 else f"(?:{'|'.join(branches)})"

    return [
        "re:^" + re.escape(prefix) + pattern(tree, r"\." if prefix else "") + r"\Z"
        for prefix, tree in trees.items()
    ]


_COMPACT_IGNORE_SUFFIX = r"\Z(?#ct-exact)"


def _expand_checkpoint_ignore(target: str) -> list[str]:
    """Decode only our finite generated regexes; user regexes remain unchanged."""
    if not (target.startswith("re:^") and target.endswith(_COMPACT_IGNORE_SUFFIX)):
        return [target]
    tokens = iter(
        re.findall(
            r"\\[\s\S]|\(\?:|\[[0-9-]+\]|.",
            target[4 : -len(_COMPACT_IGNORE_SUFFIX)],
            re.DOTALL,
        )
    )

    def expand(nested=False):
        names, alternatives = [""], []
        for token in tokens:
            if token == ")" and nested:
                return alternatives + names
            if token == "|":
                alternatives.extend(names)
                names = [""]
                continue
            if token == "(?:":
                parts = expand(True)
            elif token.startswith("["):
                parts = [
                    str(digit)
                    for span in re.findall(r"[0-9]-[0-9]|[0-9]", token)
                    for digit in range(int(span[0]), int(span[-1]) + 1)
                ]
            elif token.startswith("\\") or token not in ".^$*+?{}[]()":
                parts = [token.removeprefix("\\")]
            else:
                raise ValueError(f"Invalid generated ignore pattern: {target}")
            names = [name + part for name in names for part in parts]
        if nested:
            raise ValueError(f"Unclosed generated ignore pattern: {target}")
        return alternatives + names

    return expand()


def _compact_checkpoint_ignores(ignores: list[str]) -> list[str]:
    """Compact exact indexed names only when smaller, keeping other ignores literal."""
    groups = defaultdict(list)
    for target in ignores:
        for name in _expand_checkpoint_ignore(target):
            prefix = re.split(r"\.\d+(?:\.|$)", name, maxsplit=1)[0]
            groups[prefix].append(name)
    result = []
    for names in groups.values():
        names = list(dict.fromkeys(names))
        if len(names) > 1 and not any(name.startswith("re:") for name in names):
            compact = [
                target.removesuffix(r"\Z") + _COMPACT_IGNORE_SUFFIX
                for target in _compact_checkpoint_targets(names)
            ]
            if len(json.dumps(compact)) < len(json.dumps(names)):
                names = compact
        result.extend(names)
    return result


class QuantizationStatus(str, Enum):
    """
    Enum storing the different states a quantized layer can be in

    - Initialized: Quantization parameters are initialized to empty values
    - Calibration: Quantization parameters are being calibrated, observers are attached
    - Frozen: Quantization parameters are fully calibrated, observers are removed
    - Compressed: All parameters are quantized to target dtype. If the weight param is
        no longer applicable (i.e. if "weight" has been converted to "weight_packed"),
        the weight param is pruned from the module.
    - Decompressed: Parameters are converted back into frozen state. Quantization params
        remain on the module so that the module can be compressed, but params are
        pruned if they are no longer applicable or needed for compression
        (i.e. if "weight_packed" has been converted to "weight").
        Additionally, weight qdq is skipped during forward passes for better performance
    """

    INITIALIZED = "initialized"
    CALIBRATION = "calibration"
    FROZEN = "frozen"
    COMPRESSED = "compressed"
    DECOMPRESSED = "decompressed"

    @classmethod
    def lifecycle_order(cls) -> list["QuantizationStatus"]:
        """
        :return: list of correct quantization lifecycle order
        """
        return

    def __ge__(self, other):
        if other is None:
            return True
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) >= LIFECYCLE_ORDER.index(other)

    def __gt__(self, other):
        if other is None:
            return True
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) > LIFECYCLE_ORDER.index(other)

    def __lt__(self, other):
        if other is None:
            return False
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) < LIFECYCLE_ORDER.index(other)

    def __le__(self, other):
        if other is None:
            return False
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) <= LIFECYCLE_ORDER.index(other)


LIFECYCLE_ORDER = [
    QuantizationStatus.INITIALIZED,
    QuantizationStatus.CALIBRATION,
    QuantizationStatus.FROZEN,
    QuantizationStatus.COMPRESSED,
    QuantizationStatus.DECOMPRESSED,
]

DEFAULT_QUANTIZATION_METHOD = "compressed-tensors"
DEFAULT_QUANTIZATION_FORMAT = "fakequant"


class QuantizationConfig(BaseModel):
    """
    Full configuration specifying how a model is quantized. Each quantized layer is
    mapped to a QuantizationScheme in config_groups.

    :param config_groups: dict of QuantizationSchemes specifying the quantization
    settings for each quantized layer. A group could also be a reference to
    a predefined scheme name, mapped to a list of its target layers/classes
    :param quant_method: a constant used to differentiate compressed-tensors
    quantization from other quantization configs
    :param format: specifies how the quantized model is stored on disk
    :quantization_status: specifies the current status of all quantized layers. It is
        assumed all layers are in the same state.
    :param kv_cache_scheme: optional QuantizationArgs, that specify the
        quantization of the kv cache. If None, kv cache is not quantized.
        When applying kv cache quantization to transformer AutoModelForCausalLM,
        the kv_cache_scheme gets converted into a QuantizationScheme that:
            - targets the `q_proj` and `k_proj` modules of the model. The outputs
              of those modules are the keys and values that might be cached
            - quantizes the outputs of the aforementioned layers, so that
              keys and values are compressed before storing them in the cache
        There is an explicit assumption that the model contains modules with
        `k_proj` and `v_proj` in their names. If this is not the case
        and kv_cache_scheme != None, the quantization of kv cache will fail
    :global_compression_ratio: optional informational config to report the model
        compression ratio achieved by the quantization config
    :ignore: optional list of layers to ignore from config_groups. Layers in this list
        are not quantized even if they match up with a target in config_groups
    """

    config_groups: dict[str, QuantizationScheme | list[str]]
    quant_method: str = DEFAULT_QUANTIZATION_METHOD
    kv_cache_scheme: QuantizationArgs | None = None
    format: str = DEFAULT_QUANTIZATION_FORMAT
    quantization_status: QuantizationStatus = QuantizationStatus.INITIALIZED
    global_compression_ratio: float | None = None
    ignore: list[str] | None = Field(default_factory=list)

    @field_validator("ignore")
    @classmethod
    def _expand_ignore(cls, ignores):
        if ignores is None:
            return None
        return [
            name for target in ignores for name in _expand_checkpoint_ignore(target)
        ]

    @field_serializer("ignore")
    def _serialize_ignore(self, ignores):
        return _compact_checkpoint_ignores(ignores) if ignores is not None else None

    def model_post_init(self, __context):
        """
        updates any quantization schemes defined as presets to be fully loaded
        schemes
        """
        for group_name, targets_or_scheme in self.config_groups.items():
            if isinstance(targets_or_scheme, QuantizationScheme):
                continue  # scheme already defined
            self.config_groups[group_name] = preset_name_to_scheme(
                name=group_name,
                targets=targets_or_scheme,
            )

    def to_dict(self):
        # for compatibility with HFQuantizer
        return self.model_dump()

    @staticmethod
    def from_pretrained(
        model: Module, format: str | list | None = None
    ) -> "QuantizationConfig | None":
        """
        Converts a model into its associated QuantizationConfig based on the
        QuantizationScheme attached to each quantized module

        :param model: model to calculate quantization scheme of
        :return: filled out QuantizationScheme for the input model
        """
        from compressed_tensors.modeling import IMPL_ATTR
        from compressed_tensors.quantization.lifecycle.initialize import (
            is_cached_attention_module,
        )

        # set of all quantization schemes
        # TODO: make quant config/scheme/args frozen/hashable and use a set
        quantization_schemes: list[QuantizationScheme] = list()
        scheme_indices: dict[int, int] = {}

        # use any status from modules (in practice, use the last module)
        model_status = None

        # set of all quantized types
        # this is later used to create the ignore list
        quantization_type_names: set[str] = set()

        # maps types to names which are not quantized
        # this is later used to create the ignore list
        ignore: dict[str, list[str]] = defaultdict(list)

        # this keeps track of any kvcache schemes
        kv_cache_scheme: QuantizationArgs | None = None

        has_weight_conversions = bool(getattr(model, "_weight_conversions", None))
        named_modules = {}
        quantized_names = []
        for name, submodule in model.named_modules():
            layer_type: str = get_vllm_module_type(type(submodule).__name__)
            is_cached_attention = is_cached_attention_module(submodule)
            is_quantized = is_module_quantized(submodule)
            # Retain module references only when checkpoint names can differ.
            if has_weight_conversions:
                named_modules[name] = submodule
                if is_quantized:
                    quantized_names.append(name)

            # add config group if quantized non-attention or attention quant
            has_config_group = is_quantized and (
                not is_cached_attention or hasattr(submodule, IMPL_ATTR)
            )
            # only add kvcache if quant attention (which always implies kvcache)
            has_kv_cache = is_quantized and is_cached_attention

            if has_config_group:
                # add to running set of schemes/layer_type_names
                model_status = getattr(submodule, "quantization_status", model_status)
                quantization_type_names.add(layer_type)
                scheme = submodule.quantization_scheme
                # Resolve shared schemes once, but still merge equal copies.
                if id(scheme) not in scheme_indices:
                    try:
                        scheme_index = quantization_schemes.index(scheme)
                    except ValueError:
                        scheme_index = len(quantization_schemes)
                        quantization_schemes.append(scheme)
                    scheme_indices[id(scheme)] = scheme_index

            if has_kv_cache:
                model_status = getattr(submodule, "quantization_status", model_status)
                kv_cache_scheme = submodule.quantization_scheme.input_activations

            if not has_config_group:
                # add non-quantized layers to the ignore list
                ignore[layer_type].append(name)

        if (
            len(quantization_schemes) == 0 and kv_cache_scheme is None
        ):  # No quantized layers
            return None

        # create ignore list, only include layers whose class has ever been targeted
        consolidated_ignore = []
        for layer_type, ignore_names in ignore.items():
            if layer_type in quantization_type_names:
                # specific layers of a quantized type are ignored
                consolidated_ignore += ignore_names
            # else we leave it off the ignore list, doesn't fall under any of the
            # existing quantization schemes so it won't be quantized

        class_targets = {
            cls.__name__
            for module_type in {type(module) for module in named_modules.values()}
            for cls in module_type.__mro__
            if issubclass(cls, Module)
        }
        class_targets.difference_update(named_modules)
        checkpoint_names = {}

        def map_and_check_targets(names: list[str]) -> bool:
            mapped = _map_to_checkpoint_names(model, names)
            checkpoint_names.update(zip(names, mapped))
            return any(
                match_targets(name, named_modules[name], scheme.targets)
                != match_targets(saved, named_modules[name], scheme.targets)
                for name, saved in zip(names, mapped)
                if name != saved
                for scheme in quantization_schemes
            )

        names = list(named_modules) if has_weight_conversions else consolidated_ignore
        # Class names can also be exact saved names: keep the full collision check.
        if has_weight_conversions and not any(
            set(scheme.targets).issubset(class_targets)
            for scheme in quantization_schemes
        ):
            names = list(dict.fromkeys([*quantized_names, *consolidated_ignore]))
        remap_targets = map_and_check_targets(names)
        # Once all name-based groups need rebuilding, unrelated names cannot change
        # their assignments. Otherwise check for unquantized modules entering targets.
        if not remap_targets and len(checkpoint_names) < len(named_modules):
            remap_targets = map_and_check_targets(
                [name for name in named_modules if name not in checkpoint_names]
            )
        class_targets.difference_update(checkpoint_names.values())
        consolidated_ignore = [checkpoint_names[name] for name in consolidated_ignore]

        # Group assignments once, only when target conversion needs them.
        if remap_targets:
            scheme_names = [[] for _ in quantization_schemes]
            for name in quantized_names:
                assigned = named_modules[name].quantization_scheme
                if id(assigned) not in scheme_indices:
                    # KV-cache-only modules may share an equal config-group scheme.
                    try:
                        assigned_index = quantization_schemes.index(assigned)
                    except ValueError:
                        assigned_index = -1
                    scheme_indices[id(assigned)] = assigned_index
                assigned_index = scheme_indices[id(assigned)]
                if assigned_index >= 0:
                    scheme_names[assigned_index].append(name)

        # create config groups from all unique schemes
        config_groups = {}
        for idx, scheme in enumerate(quantization_schemes):
            # Rebuild name-based groups together: replacing exact names with regexes
            # must not change precedence against another group's retained regex.
            # Class-only targets remain lower-priority defaults.
            if remap_targets and not set(scheme.targets).issubset(class_targets):
                scheme = scheme.model_copy(
                    update={
                        "targets": _compact_checkpoint_targets(
                            [checkpoint_names[name] for name in scheme_names[idx]]
                        )
                    }
                )
            group_name = "group_" + str(idx)
            config_groups[group_name] = scheme

        # infer format
        if format is None:
            if model_status == QuantizationStatus.COMPRESSED:
                format = CompressionFormat.int_quantized.value
            else:
                format = CompressionFormat.dense.value
        elif isinstance(format, list):
            format = (
                CompressionFormat.mixed_precision.value
                if len(format) > 1
                else format[0]
            )

        return QuantizationConfig(
            config_groups=config_groups,
            quantization_status=model_status,
            kv_cache_scheme=kv_cache_scheme,
            global_compression_ratio=None,
            format=format,
            ignore=consolidated_ignore,
        )

    def requires_calibration_data(self):
        if self.kv_cache_scheme is not None:
            return True

        for _, scheme in self.config_groups.items():
            if scheme.weights is not None:
                if scheme.weights.observer == "imatrix_mse":
                    return True
            if scheme.input_activations is not None:
                if scheme.input_activations.dynamic in (False, DynamicType.LOCAL):
                    return True
            if scheme.output_activations is not None:
                if not scheme.output_activations.dynamic:
                    return True

        return False

    def merge(self, config: "QuantizationConfig") -> None:
        """
        Merge another QuantizationConfig into self, modifying in-place. The current
        quant config (self) will take precedence over the second quant config,
        i.e. the config groups will be appended rather than inserted before. Python's
        json stdlib and pydantic both respect order when serializing/deserializing.

        Because QuantizationConfig has a global ignore list while targets are scoped to
        a given scheme, merging is not a straightforward task. A warning will be raised
        when this function is called to indicate that invalid quantization configs are
        possible when using this method, particularly when complex ignore lists are
        used. For best results, use complex targets lists over complex ignore lists.

        If a new quant format is added, the global format will be set to
        "mixed-precision".

        The QuantizationStatus will be set to whichever is largest.

        Excluding regexes, any names in the ignore list that are targeted in the new
        quant config will be pruned.

        :param config: QuantizationConfig to merge into self.
        """
        warnings.warn(
            "Attempting to merge quantization configs. This is not a straightforward "
            "task and can lead to quantization configs that fail to load. For best "
            "results, use complex targets lists instead of complex ingore lists"
        )

        pruned_ignore_list = []
        for ign in self.ignore:
            if ign.startswith("re:"):
                pruned_ignore_list.append(ign)
                continue
            if any(
                match_name(ign, target)
                for scheme in config.config_groups.values()
                for target in scheme.targets
            ):
                continue
            pruned_ignore_list.append(ign)
        self.ignore = pruned_ignore_list

        for scheme_name, scheme in config.config_groups.items():
            new_scheme_name = find_unique_name(scheme_name, self.config_groups.keys())

            self.config_groups[new_scheme_name] = scheme

        unique_formats = set(scheme.format for scheme in self.config_groups.values())
        self.format = (
            next(iter(unique_formats))
            if len(unique_formats) == 1
            else CompressionFormat.mixed_precision.value
        )

        if config.quantization_status > self.quantization_status:
            self.quantization_status = config.quantization_status

    # TODO set `extra="forbid"` when upstream transformers is compatible
    model_config = ConfigDict(extra="ignore")


def get_vllm_module_type(module_type: str) -> str:
    """
    Returns a string representing the module type used when loading in vLLM.
    This is typically going to be the same as the `torch.nn.Module` type,
    however specific cases like MoE gate layers need to be treated like "Linear"
    layers for the purposes of config matching.
    """
    if "ExpertMLP" not in module_type and (
        "Router" in module_type
        or "Gating" in module_type
        or re.search(r"Gate(?![a-z])", module_type)
    ):
        module_type = "Linear"

    return module_type
