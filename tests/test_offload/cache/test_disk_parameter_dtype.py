# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.offload import from_accelerate, offload_module, to_accelerate
from compressed_tensors.offload.cache import DiskCache
from compressed_tensors.offload.module import (
    remove_module_offload,
    stage_module_offload,
)
from safetensors.torch import load_file, save_file
from transformers import PretrainedConfig, PreTrainedModel


@pytest.fixture(autouse=True)
def isolated_disk_index(monkeypatch):
    monkeypatch.setattr(DiskCache, "index", {})


@pytest.mark.unit
@pytest.mark.parametrize("method", ["onload", "stage"])
@pytest.mark.parametrize("checkpoint_dtype", [torch.float32, torch.bfloat16])
def test_disk_read_preserves_parameter_after_dtype_cast(
    tmp_path, method, checkpoint_dtype
):
    parameter = torch.nn.Parameter(
        torch.arange(16, dtype=torch.bfloat16).reshape(4, 4), requires_grad=False
    )
    parameter.test_metadata = "preserve"
    cache = DiskCache("cpu", offload_dir=tmp_path)
    offloaded = cache.offload(parameter)
    entry = DiskCache.index[offloaded]
    save_file(
        {"weight": parameter.data.to(checkpoint_dtype)}, entry["safetensors_file"]
    )

    onloaded = getattr(cache, method)(offloaded)

    assert isinstance(onloaded, torch.nn.Parameter)
    assert onloaded.dtype == parameter.dtype
    assert onloaded.requires_grad == parameter.requires_grad
    assert onloaded.test_metadata == parameter.test_metadata
    assert torch.equal(onloaded, parameter)


class _OffloadedModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.q_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)


@pytest.mark.unit
@pytest.mark.parametrize("save_before_restore", [False, True])
def test_dtype_cast_staging_supports_save_and_offload_restore(
    tmp_path, save_before_restore
):
    pytest.importorskip("accelerate")
    model = _OffloadedModel(PretrainedConfig())
    reference = model.q_proj.weight.detach().clone()
    offload_dir = tmp_path / "offload"
    offload_dir.mkdir()
    offload_module(
        model.q_proj,
        onload_device="cpu",
        offload_device="disk",
        offload_dir=offload_dir,
    )
    template = model.q_proj._parameters.offloaded_values["weight"]
    entry = DiskCache.index[template]
    save_file({"weight": reference.float()}, entry["safetensors_file"])

    stage_module_offload(model.q_proj)
    remove_module_offload(model.q_proj, onload_tensors=True)
    offload_module(
        model.q_proj,
        onload_device="cpu",
        offload_device="disk",
        offload_dir=offload_dir,
    )
    to_accelerate(model)

    if save_before_restore:
        # Source-rank saving must resolve meta state dict entries as Parameters.
        model.save_pretrained(tmp_path / "saved")
        saved = load_file(tmp_path / "saved" / "model.safetensors")
        assert torch.equal(saved["q_proj.weight"], reference)
    # Other ranks restore CT offloading without assigning plain tensors as params.
    from_accelerate(model)
    assert isinstance(model.q_proj.weight, torch.nn.Parameter)
    assert torch.equal(model.q_proj.weight, reference)
