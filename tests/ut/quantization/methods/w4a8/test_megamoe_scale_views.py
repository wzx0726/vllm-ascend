# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the private MegaMoe weight getter, including shared-expert fusion."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import WeakKeyDictionary

import pytest
import torch

ROOT = Path(__file__).resolve().parents[5] / "vllm_ascend"
MXFP4 = ROOT / "quantization/methods/w4a8/w4a8_mxfp4.py"
WEIGHTS = ROOT / "ops/fused_moe/dataclass/fused_experts.py"
ROUTED_GROUPS = (("w13_weight_scale_list", "w1_scale"), ("w2_weight_scale_list", "w2_scale"))


def _class(path, name):
    return next(
        node for node in ast.parse(path.read_text(encoding="utf-8")).body if getattr(node, "name", None) == name
    )


@pytest.fixture
def implementation(monkeypatch):
    # Load the source methods without importing the NPU runtime and distributed plugin.
    ctx = SimpleNamespace(use_mega_moe=True)
    config = SimpleNamespace(
        additional_config={"megamoe_checked_scale_views": True},
        compilation_config=SimpleNamespace(cudagraph_mode="NONE"),
        quant_config=SimpleNamespace(quant_description={}),
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(enable_eplb=True),
    )
    ns = dict(
        torch=torch,
        dataclass=dataclass,
        _EXTRA_CTX=ctx,
        _MEGAMOE_SCALE_VIEWS=WeakKeyDictionary(),
        CUDAGraphMode=SimpleNamespace(NONE="NONE"),
        get_current_vllm_config=lambda: config,
        get_ascend_config=lambda: SimpleNamespace(),
    )
    scheme = _class(MXFP4, "AscendW4A8MXFPDynamicFusedMoEMethod")
    scheme.bases = scheme.decorator_list = []
    scheme.body = [
        method
        for method in scheme.body
        if isinstance(method, ast.FunctionDef)
        and method.name in {"__init__", "_get_cann_mega_moe_weights", "get_fused_mc2_weights", "_get_mlp_weights"}
    ]
    ds = _class(MXFP4, "AscendW4A8MXFPDSDynamicFusedMoEMethod")
    ds.decorator_list = []
    ds.body = [ast.copy_location(ast.Pass(), ds)]
    module = ast.Module(body=[_class(WEIGHTS, "MoEWeights"), scheme, ds], type_ignores=[])
    exec(compile(module, str(MXFP4), "exec"), ns)

    # For these ordinary contiguous CPU tensors, is_set_to checks storage, offset and layout.
    def matches(first, second, views1, views2):
        return all(
            len(group) == len(views) and all(t.is_set_to(v) for t, v in zip(group, views))
            for group, views in ((first, views1), (second, views2))
        )

    guard = Mock(side_effect=matches)
    ops = SimpleNamespace(_mxfp4_scale_views_match=guard)
    monkeypatch.setattr(torch.ops, "_C_ascend", ops)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), raising=False)
    return ns[scheme.name], ctx, config, ops, guard


def _layer(experts=2):
    layer = torch.nn.Module()
    layer.w13_weight_list = [torch.zeros(2, 2) for _ in range(experts)]
    layer.w2_weight_list = [torch.zeros(2, 2) for _ in range(experts)]
    for name, _ in ROUTED_GROUPS:
        setattr(layer, name, [torch.full((2, 2), 127, dtype=torch.uint8) for _ in range(experts)])
    return layer


@pytest.mark.parametrize(("source", "field"), ROUTED_GROUPS)
@torch.inference_mode()
def test_private_scale_views_follow_current_storage(implementation, source, field):
    cls, _, _, _, _ = implementation
    get = cls()._get_cann_mega_moe_weights
    layer = _layer(experts=128)
    first, warm = get(layer), get(layer)
    for _, attr in ROUTED_GROUPS:
        left, right = getattr(first, attr), getattr(warm, attr)
        assert left is not right
        assert all(a is b for a, b in zip(left, right, strict=True))
    inputs = getattr(layer, source)
    held = getattr(first, field)[0]
    assert held.float()[0, 0].item() == 1
    inputs[0].copy_(torch.full_like(inputs[0], 128))
    assert getattr(get(layer), field)[0] is held
    assert held.float()[0, 0].item() == 2
    inputs[0] = torch.full_like(inputs[0], 129)
    replaced = getattr(get(layer), field)[0]
    assert replaced.float()[0, 0].item() == 4
    inputs[0].data = torch.full_like(inputs[0], 130)
    assert getattr(get(layer), field)[0].float()[0, 0].item() == 8
    assert held.float()[0, 0].item() == 2
    assert replaced.float()[0, 0].item() == 4


@torch.inference_mode()
def test_warm_routed_cache_preserves_current_shared_expert(implementation):
    cls, ctx, _, _, guard = implementation
    path = ROOT / "ops/fused_moe/moe_comm_method.py"
    comm = _class(path, "FusedMC2CommImpl")
    comm.bases = []
    comm.body = [node for node in comm.body if getattr(node, "name", None) == "fused_experts"]
    ctx.is_decode_only_node = False
    ns = dict(
        torch=torch,
        _EXTRA_CTX=ctx,
        FusedExpertsResult=SimpleNamespace,
        MoEFusedExpertsInput=SimpleNamespace,
        TokenDispatcherWithMC2=SimpleNamespace,
    )
    exec(compile(ast.Module(body=[comm], type_ignores=[]), str(path), "exec"), ns)
    runner = ns["FusedMC2CommImpl"]()
    runner.enable_fused_mc2 = 1
    runner.token_dispatcher = SimpleNamespace()
    runner._apply_cann_mega_moe = lambda inputs, weights, **kwargs: (weights, None)
    method, layer = cls(), _layer()
    request = SimpleNamespace(layer=layer)
    first = runner.fused_experts(request, method).routed_out
    warm = runner.fused_experts(request, method).routed_out
    assert first.w1_scale[0] is warm.w1_scale[0]
    guard.reset_mock()
    layer.ascend_megamoe_shared_fusion = SimpleNamespace()
    for generation in (1, 2):
        shared = SimpleNamespace(**{k: [torch.tensor(generation)] for k in ("w1", "w2", "w1_scale", "w2_scale")})
        layer.ascend_megamoe_shared_fusion.weights = shared
        result = runner.fused_experts(request, method).routed_out
        for name in vars(shared):
            assert getattr(result, "shared_" + name) is getattr(shared, name)
    guard.assert_not_called()


@pytest.mark.parametrize("mode", ["default", "old_extension", "mega_moe_off"])
@torch.inference_mode()
def test_original_getter_fallbacks(implementation, mode):
    cls, ctx, config, ops, guard = implementation
    if mode == "default":
        config.additional_config.clear()
    if mode == "old_extension":
        del ops._mxfp4_scale_views_match
    if mode == "mega_moe_off":
        ctx.use_mega_moe = False
    method = cls()
    layer = _layer()
    reference = method.get_fused_mc2_weights(layer)
    first, second = method._get_cann_mega_moe_weights(layer), method._get_cann_mega_moe_weights(layer)
    guard.assert_not_called()
    for _, field in ROUTED_GROUPS:
        expected, a, b = (getattr(value, field)[0] for value in (reference, first, second))
        assert a is not b
        assert expected.dtype == a.dtype and expected.is_set_to(a)
