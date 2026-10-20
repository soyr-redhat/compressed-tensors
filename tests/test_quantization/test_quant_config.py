# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.quantization import (
    DEFAULT_QUANTIZATION_FORMAT,
    DEFAULT_QUANTIZATION_METHOD,
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
)
from compressed_tensors.quantization.quant_config import (
    _compact_checkpoint_ignores,
    _compact_checkpoint_targets,
    _map_to_checkpoint_names,
    get_vllm_module_type,
)
from compressed_tensors.quantization.quant_scheme import preset_name_to_scheme
from compressed_tensors.utils.match import match_name, match_named_modules
from pydantic import ValidationError
from transformers import AutoModelForImageTextToText
from transformers.core_model_loading import WeightRenaming


def test_basic_config():
    config_groups = {"group_1": QuantizationScheme(targets=[])}
    config = QuantizationConfig(config_groups=config_groups)

    assert config.config_groups == config_groups
    assert config.quant_method == DEFAULT_QUANTIZATION_METHOD
    assert config.format == DEFAULT_QUANTIZATION_FORMAT
    assert config.quantization_status == QuantizationStatus.INITIALIZED
    assert config.global_compression_ratio is None
    assert isinstance(config.ignore, list) and len(config.ignore) == 0
    assert config.moe_sparsity_config is None


def test_config_with_moe_sparsity():
    from compressed_tensors.config import ExpertSparsityConfig, MoESparsityMethod

    config = QuantizationConfig(
        config_groups={"group_1": QuantizationScheme(targets=[])},
        moe_sparsity_config=ExpertSparsityConfig(
            method="prune", num_experts=[None, 129, 253]
        ),
    )
    assert isinstance(config.moe_sparsity_config, ExpertSparsityConfig)
    assert config.moe_sparsity_config.method == MoESparsityMethod.PRUNE

    # round-trips through dict (de)serialization
    restored = QuantizationConfig.model_validate(config.model_dump())
    assert isinstance(restored.moe_sparsity_config, ExpertSparsityConfig)
    assert restored.moe_sparsity_config.num_experts == [None, 129, 253]


def test_full_config():
    config_groups = {
        "group_1": QuantizationScheme(targets=[]),
        "group_2": QuantizationScheme(targets=[]),
    }
    global_compression_ratio = 3.5
    ignore = ["model.layers.0"]
    quantization_status = "compressed"

    config = QuantizationConfig(
        config_groups=config_groups,
        global_compression_ratio=global_compression_ratio,
        ignore=ignore,
        quantization_status=quantization_status,
    )
    assert config.config_groups == config_groups
    assert config.global_compression_ratio == global_compression_ratio
    assert config.ignore == ignore
    assert config.quantization_status == QuantizationStatus.COMPRESSED


def test_need_config_groups():
    with pytest.raises(ValidationError):
        _ = QuantizationScheme()


@pytest.mark.parametrize(
    "scheme_name",
    ["W8A8", "W8A16", "W4A16", "FP8"],
)
def test_load_scheme_from_preset(scheme_name: str):
    targets = ["Linear"]
    config = QuantizationConfig(config_groups={scheme_name: targets})

    assert scheme_name in config.config_groups
    assert isinstance(config.config_groups[scheme_name], QuantizationScheme)
    assert config.config_groups[scheme_name].targets == targets


def test_to_dict():
    """Test serialization of QuantizationConfig including format"""

    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(num_bits=4, symmetric=True, group_size=128),
        ),
        "group_2": QuantizationScheme(
            targets=["Conv2d"],
            weights=QuantizationArgs(num_bits=8),
        ),
    }
    config = QuantizationConfig(
        config_groups=config_groups,
        global_compression_ratio=3.5,
        ignore=["model.layers.0"],
        quantization_status="compressed",
        format="int-quantized",
    )

    # Serialize to dict
    config_dict = config.to_dict()
    assert "config_groups" in config_dict
    assert config_dict["format"] == "int-quantized"
    assert config_dict["quantization_status"] == "compressed"

    # Deserialize from dict
    reloaded = QuantizationConfig.model_validate(config_dict)
    assert config == reloaded


@pytest.mark.parametrize(
    "model_id,hf_ignores,checkpoint_ignores",
    [
        pytest.param(
            "llava-hf/llava-interleave-qwen-0.5b-hf",
            [
                "lm_head",
                "model.vision_tower.encoder.layers.0.self_attn.q_proj",
                "model.multi_modal_projector.linear_1",
            ],
            [
                "lm_head",
                "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
                "multi_modal_projector.linear_1",
            ],
            id="llava",
        ),
        pytest.param(
            "google/gemma-4-12b-it",
            [
                "lm_head",
                "model.embed_vision.patch_dense",
                "model.embed_vision.multimodal_embedder.embedding_projection",
            ],
            [
                "lm_head",
                "model.vision_embedder.patch_dense",
                "model.embed_vision.embedding_projection",
            ],
            id="gemma4",
        ),
        pytest.param(
            "Qwen/Qwen2-VL-2B-Instruct",
            [
                "lm_head",
                "model.visual.merger.mlp.0",
            ],
            [
                "lm_head",
                "visual.merger.mlp.0",
            ],
            id="qwen2_vl",
        ),
    ],
)
def test_map_to_checkpoint_names(model_id, hf_ignores, checkpoint_ignores):
    """Load a real model and verify that HF module names in the ignore list
    are reverse-mapped to checkpoint key names.

    Uses ``device_map="meta"`` so no weights are materialised -- only the
    model structure and its ``_weight_conversions`` are needed.
    """
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.float16, device_map="meta"
    )

    result = _map_to_checkpoint_names(model, hf_ignores)

    assert result == checkpoint_ignores


def _make_model(names, conversions=()):
    model = torch.nn.Module()
    for name in names:
        parent = model
        *path, leaf = name.split(".")
        for part in path:
            if part not in parent._modules:
                parent.add_module(part, torch.nn.Module())
            parent = parent.get_submodule(part)
        parent.add_module(leaf, torch.nn.Linear(4, 4))
    model._weight_conversions = [WeightRenaming(*pair) for pair in conversions]
    return model


def _attach_scheme(model, names, targets, bits=8):
    scheme = QuantizationScheme(
        targets=targets, weights=QuantizationArgs(num_bits=bits)
    )
    for name in names:
        module = model.get_submodule(name)
        module.quantization_scheme = scheme
        module.quantization_status = QuantizationStatus.FROZEN
    return scheme


@pytest.mark.parametrize(
    "targets",
    [["runtime.0"], [r"re:^runtime\."], ["Linear"], ["Linear", r"re:^runtime\."]],
)
def test_export_targets_uses_checkpoint_names_without_mutating_scheme(targets):
    model = _make_model(
        ["runtime.0", "runtime.1"],
        [
            ("checkpoint.projection.weight", "runtime.0.weight"),
            ("checkpoint.ignored.weight", "runtime.1.weight"),
        ],
    )
    mixed = len(targets) == 2
    scheme = _attach_scheme(
        model, ["runtime.0", "runtime.1"] if mixed else ["runtime.0"], targets
    )
    exported = QuantizationConfig.from_pretrained(model)
    expected = (
        [r"re:^checkpoint\.(?:ignored|projection)\Z"]
        if mixed
        else ["Linear"]
        if targets == ["Linear"]
        else ["checkpoint.projection"]
    )
    assert exported == QuantizationConfig.from_pretrained(model)
    assert exported.config_groups["group_0"].targets == expected
    assert exported.ignore == ([] if mixed else ["checkpoint.ignored"])
    assert scheme.targets == targets


@pytest.mark.parametrize("include_linear", [False, True])
def test_export_preserves_kv_cache_assignments(include_linear):
    class CacheAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.k_proj = torch.nn.Linear(4, 4)

        def forward(self, hidden_states, past_key_value=None):
            return hidden_states

    model = torch.nn.Module()
    model.runtime = torch.nn.ModuleDict({"attention": CacheAttention()})
    model._weight_conversions = [WeightRenaming("checkpoint.", "runtime.")]
    scheme = QuantizationScheme(
        targets=[r"re:^runtime\."], input_activations=QuantizationArgs()
    )
    model.runtime["attention"].quantization_scheme = scheme.model_copy()
    model.runtime["attention"].quantization_status = QuantizationStatus.FROZEN
    if include_linear:
        model.runtime["linear"] = torch.nn.Linear(4, 4)
        model.runtime["linear"].quantization_scheme = scheme
        model.runtime["linear"].quantization_status = QuantizationStatus.FROZEN

    exported = QuantizationConfig.from_pretrained(model)

    assert exported.kv_cache_scheme == scheme.input_activations
    if include_linear:
        assert exported.config_groups["group_0"].targets == [
            r"re:^checkpoint\.(?:attention|linear)\Z",
        ]
    else:
        assert exported.config_groups == {}


@pytest.mark.parametrize(
    "target,collision", [(r"re:^runtime\.", "runtime.saved"), ("Linear", "Linear")]
)
def test_export_excludes_unassigned_target_collisions(target, collision):
    prefix = "checkpoint" if target == "Linear" else "runtime"
    model = _make_model(
        ["runtime.0", "runtime.1"],
        [(prefix + ".", "runtime."), (collision + ".weight", "other.weight")],
    )
    model.other = torch.nn.Embedding(4, 4)
    _attach_scheme(model, ["runtime.0"], [target])
    _attach_scheme(model, ["runtime.1"], ["runtime.1"], 4)
    exported = QuantizationConfig.from_pretrained(model)
    # Keeping the broad target would quantize an unassigned embedding on reload.
    assert exported.ignore == []
    assert [g.targets for g in exported.config_groups.values()] == [
        [f"{prefix}.0"],
        [f"{prefix}.1"],
    ]
    checkpoint = _make_model([f"{prefix}.0", f"{prefix}.1", collision])
    parent, _, leaf = collision.rpartition(".")
    checkpoint.get_submodule(parent).add_module(leaf, torch.nn.Embedding(4, 4))
    apply_quantization_config(checkpoint, exported)
    for index, bits in enumerate((8, 4)):
        module = checkpoint.get_submodule(f"{prefix}.{index}")
        assert module.quantization_scheme.weights.num_bits == bits
    assert not hasattr(checkpoint.get_submodule(collision), "quantization_scheme")


@pytest.mark.parametrize("renamed", [False, True])
def test_export_skips_unnecessary_name_mapping(monkeypatch, renamed):
    from compressed_tensors.quantization import quant_config

    model = _make_model(
        [f"runtime.{i}" for i in range(3)],
        [("checkpoint.", "runtime.")] if renamed else [],
    )
    model.other = torch.nn.Embedding(4, 4)
    for index, bits in enumerate((4, 8)):
        _attach_scheme(model, [f"runtime.{index}"], [f"runtime.{index}"], bits)
    mapped_names = []

    def record_mapping(model, names):
        mapped_names.extend(names)
        return _map_to_checkpoint_names(model, names)

    monkeypatch.setattr(quant_config, "_map_to_checkpoint_names", record_mapping)
    exported = QuantizationConfig.from_pretrained(model)
    assert mapped_names == (
        ["runtime.0", "runtime.1", "runtime.2"] if renamed else ["runtime.2"]
    )
    prefix = "checkpoint" if renamed else "runtime"
    assert exported.ignore == [f"{prefix}.2"]
    for index, bits in enumerate((4, 8)):
        group = exported.config_groups[f"group_{index}"]
        assert group.targets == [f"{prefix}.{index}"]
        assert group.weights.num_bits == bits
    # A later export must observe newly assigned modules and changed conversions.
    _attach_scheme(model, ["other"], ["other"])
    model._weight_conversions = [WeightRenaming("saved.", "runtime.")]
    exported = QuantizationConfig.from_pretrained(model)
    assert exported.ignore == ["saved.2"]
    assert exported.config_groups["group_0"].targets == ["saved.0"]
    assert exported.config_groups["group_2"].targets == ["other"]


@pytest.mark.parametrize(
    "targets", [["attention.0"], [r"re:^attention\."], ["Linear", r"re:^attention\."]]
)
def test_export_preserves_targets_with_unrelated_conversions(targets):
    model = _make_model(
        [f"attention.{i}" for i in range(64)],
        [("checkpoint.expert.weight", "expert.weight")],
    )
    model.expert = torch.nn.Embedding(4, 4)
    scheme = _attach_scheme(
        model, [name for name, _ in match_named_modules(model, targets)], targets
    )
    exported = QuantizationConfig.from_pretrained(model)
    assert exported == QuantizationConfig.from_pretrained(model)
    assert exported.config_groups["group_0"].targets == targets
    assert scheme.targets == targets


def test_export_preserves_regex_when_renamed_modules_still_match():
    model = _make_model(
        ["attention.runtime"], [("attention.saved.weight", "attention.runtime.weight")]
    )
    scheme = _attach_scheme(model, ["attention.runtime"], [r"re:^attention\."])
    exported = QuantizationConfig.from_pretrained(model)
    assert exported.config_groups["group_0"].targets == scheme.targets
    checkpoint = _make_model(["attention.saved"])
    apply_quantization_config(checkpoint, exported)
    assert checkpoint.attention.saved.quantization_scheme.weights == scheme.weights


def test_export_preserves_groups_when_rename_enters_another_groups_regex():
    model = _make_model(["runtime.0", "other"], [("runtime.1.weight", "other.weight")])
    _attach_scheme(model, ["runtime.0"], [r"re:^runtime\."])
    _attach_scheme(model, ["other"], ["other"], 4)
    exported = QuantizationConfig.from_pretrained(model)
    assert exported.config_groups["group_0"].targets == ["runtime.0"]
    assert exported.config_groups["group_1"].targets == ["runtime.1"]
    checkpoint = _make_model(["runtime.0", "runtime.1"])
    apply_quantization_config(checkpoint, exported)
    for index, bits in enumerate((8, 4)):
        module = checkpoint.get_submodule(f"runtime.{index}")
        assert module.quantization_scheme.weights.num_bits == bits


def test_export_preserves_exact_target_priority_over_regex_after_rename():
    model = _make_model(
        [f"runtime.{i}" for i in range(3)],
        [(f"runtime.saved{i}.weight", f"runtime.{i}.weight") for i in range(2)],
    )
    _attach_scheme(
        model, ["runtime.0", "runtime.1"], ["runtime.0", "runtime.1", r"re:^runtime\."]
    )
    _attach_scheme(model, ["runtime.2"], [r"re:^runtime\."], 4)
    exported = QuantizationConfig.from_pretrained(model)
    assert exported.config_groups["group_0"].targets == [
        r"re:^runtime\.(?:saved0|saved1)\Z"
    ]
    assert exported.config_groups["group_1"].targets == ["runtime.2"]
    checkpoint = _make_model([f"runtime.{name}" for name in ("saved0", "saved1", "2")])
    apply_quantization_config(checkpoint, exported)
    for name, bits in (("saved0", 8), ("saved1", 8), ("2", 4)):
        module = checkpoint.get_submodule(f"runtime.{name}")
        assert module.quantization_scheme.weights.num_bits == bits


@pytest.mark.parametrize(
    "names,excluded",
    [
        ([], ["layer"]),
        (["layer"], ["layers", "layer.child"]),
        (["layer", "layer.child"], ["layer.other", "layer."]),
        (["a+b.0", "a+b.1"], ["ab.0", "a+b.2"]),
        (
            ["layers.0.experts.0.up_proj", "layers.1.experts.1.down_proj"],
            ["layers.0.experts.1.down_proj", "layers.1.experts.0.up_proj"],
        ),
    ],
)
def test_compact_checkpoint_targets_matches_only_assigned_names(names, excluded):
    targets = _compact_checkpoint_targets(names)
    assert targets == _compact_checkpoint_targets(list(reversed(names)))
    for name in names:
        assert any(match_name(name, target) for target in targets)
    for name in excluded + [name + "\n" for name in names]:
        assert not any(match_name(name, target) for target in targets)


def test_compact_checkpoint_targets_large_expert_layout():
    names = [
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"
        for layer in range(64)
        for expert in range(512)
        for projection in ("gate", "up", "down")
    ]
    targets = _compact_checkpoint_targets(names)
    assert len(targets) == 1
    assert len(targets[0]) < 200
    assert all(match_name(name, targets[0]) for name in names)
    for name in (
        "model.layers.64.mlp.experts.0.up_proj",
        "model.layers.0.mlp.experts.512.up_proj",
        "model.layers.0.mlp.experts.0.other_proj",
    ):
        assert not match_name(name, targets[0])


@pytest.mark.parametrize(
    "indices",
    [
        list(map(str, range(64))),
        list(map(str, range(512))),
        ["0", "2", "9", "10", "12", "19", "99", "100"],
        ["0", "00", "01", "1", "10"],
    ],
)
def test_compact_checkpoint_targets_numeric_ranges_are_exact(indices):
    names = [f"model.layers.{index}.up_proj" for index in indices]
    targets = _compact_checkpoint_targets(names)
    assert targets == _compact_checkpoint_targets(list(reversed(names)))
    candidates = {str(i) for i in range(600)} | {f"0{i}" for i in range(100)}
    candidates.update(["-1", "1.0", "١", "１", "1\n"])
    for index in candidates:
        matched = any(match_name(f"model.layers.{index}.up_proj", t) for t in targets)
        assert matched == (index in indices), (index, targets)


def test_compact_checkpoint_targets_preserves_separate_namespace_prefixes():
    names = [
        f"{prefix}.layers.{layer}.mlp.{projection}_proj"
        for prefix in ("model", "model.mtp")
        for layer in range(2)
        for projection in ("gate", "up")
    ]
    targets = _compact_checkpoint_targets(names)
    assert len(targets) == 2
    assert targets[0].startswith(r"re:^model\.layers\.")
    assert targets[1].startswith(r"re:^model\.mtp\.layers\.")
    assert all(
        sum(match_name(name, target) for target in targets) == 1 for name in names
    )


def test_compact_checkpoint_targets_accepts_large_irregular_patterns():
    names = [f"irregular_module_{i}" for i in range(2000)]
    targets = _compact_checkpoint_targets(names)
    assert len(targets) == 1
    assert len(targets[0].encode("utf-8")) > 16_384
    assert all(match_name(name, targets[0]) for name in names)
    assert not match_name("irregular_module_2000", targets[0])


@pytest.mark.parametrize(
    "mtp_preset", ["FP8_DYNAMIC", "W4A16"], ids=["same", "different"]
)
def test_export_backbone_and_mtp_schemes(mtp_preset):
    backbone = [f"model.layers.{i}.up_proj" for i in range(2)]
    mtp = [f"mtp.layers.0.experts.{i}.up_proj" for i in range(4)]
    model = _make_model(
        [*backbone, *mtp, "lm_head"], [("model.layers.2.", "mtp.layers.0.")]
    )
    schemes = [
        preset_name_to_scheme("FP8_DYNAMIC", ["Linear"]),
        preset_name_to_scheme(mtp_preset, [r"re:^mtp\."]),
    ]
    original = [s.model_dump() for s in schemes]
    for names, scheme in zip((backbone, mtp), schemes):
        for name in names:
            module = model.get_submodule(name)
            # Equal scheme copies must still export as one group.
            module.quantization_scheme = scheme.model_copy()
            module.quantization_status = QuantizationStatus.FROZEN
    exported = QuantizationConfig.from_pretrained(model)
    assert exported == QuantizationConfig.from_pretrained(model)
    assert len(exported.config_groups) == 2
    assert exported.config_groups["group_0"].targets == ["Linear"]
    assert len(exported.config_groups["group_1"].targets) == 1
    saved_mtp = [name.replace("mtp.layers.0.", "model.layers.2.") for name in mtp]
    checkpoint = _make_model([*backbone, *saved_mtp, "lm_head"])
    apply_quantization_config(checkpoint, exported)
    for names, scheme in zip((backbone, saved_mtp), schemes):
        for name in names:
            restored = checkpoint.get_submodule(name).quantization_scheme
            assert restored.model_dump(exclude={"targets"}) == scheme.model_dump(
                exclude={"targets"}
            )
    assert not hasattr(checkpoint.lm_head, "quantization_scheme")
    assert [s.model_dump() for s in schemes] == original


def test_get_vllm_module_type():
    assert get_vllm_module_type("ExpertMLP") == "ExpertMLP"
    assert get_vllm_module_type("ExpertMLPWithGate") == "ExpertMLPWithGate"
    assert get_vllm_module_type("ExpertMLPWithoutGate") == "ExpertMLPWithoutGate"
    assert get_vllm_module_type("Linear") == "Linear"
    assert get_vllm_module_type("DeepseekV4TopKRouter") == "Linear"
    assert get_vllm_module_type("DeepseekV4HashRouter") == "Linear"
    assert get_vllm_module_type("JetMoeTopKGating") == "Linear"
    assert get_vllm_module_type("DeepseekV4TopKGate") == "Linear"
    assert get_vllm_module_type("Qwen3NextGatedDeltaNet") == "Qwen3NextGatedDeltaNet"
    assert get_vllm_module_type("Qwen3NextRMSNormGated") == "Qwen3NextRMSNormGated"


def test_quantization_config_merge():
    config = QuantizationConfig(
        config_groups={
            "config_group_0": QuantizationScheme(
                targets=["re:.*self_attn.*"],
                weights=QuantizationArgs(num_bits=4, symmetric=True, group_size=128),
            )
        },
        ignore=["lm_head", "model.layers.0.mlp.gate_proj", "re:.*mtp.*"],
        quantization_status=QuantizationStatus.INITIALIZED,
    )

    new_config = QuantizationConfig(
        config_groups={
            "config_group_0": QuantizationScheme(
                targets=["re:.*mlp.*"],
                weights=QuantizationArgs(num_bits=8, symmetric=False, group_size=128),
            )
        },
        ignore=["lm_head"],
        quantization_status=QuantizationStatus.COMPRESSED,
    )

    config.merge(new_config)

    ordered_schemes = list(config.config_groups.values())
    assert len(ordered_schemes) == 2
    assert ordered_schemes[0].targets[0] == "re:.*self_attn.*"
    assert ordered_schemes[1].targets[0] == "re:.*mlp.*"

    # should strip out "model.layers.0.mlp.gate_proj" from ignore
    assert set(config.ignore) == set(["lm_head", "re:.*mtp.*"])

    assert config.quantization_status == QuantizationStatus.COMPRESSED


def test_compact_ignore_preserves_irregular_names_and_small_lists():
    names = [f"module.layers.{i}.unique_projection_{i}" for i in range(2000)]
    config = QuantizationConfig(config_groups={}, ignore=names)
    assert set(
        QuantizationConfig.model_validate_json(config.model_dump_json()).ignore
    ) == set(names)
    assert _compact_checkpoint_ignores(["lm_head", "re:.*mtp.*"]) == [
        "lm_head",
        "re:.*mtp.*",
    ]
    assert (
        QuantizationConfig(config_groups={}, ignore=None).model_dump()["ignore"] is None
    )


def test_imatrix_mse_weight_observer_requires_calibration_data():
    from compressed_tensors.quantization import QuantizationArgs

    config = QuantizationConfig(
        config_groups={
            "group_1": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(observer="imatrix_mse"),
            )
        }
    )

    assert config.requires_calibration_data()


def test_default_weight_observer_does_not_require_calibration_data():
    from compressed_tensors.quantization import QuantizationArgs

    config = QuantizationConfig(
        config_groups={
            "group_1": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(),
            )
        }
    )

    assert not config.requires_calibration_data()
