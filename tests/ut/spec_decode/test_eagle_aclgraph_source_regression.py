# SPDX-License-Identifier: Apache-2.0
"""Source-level regressions for the verified-main Eagle ACL graph patch."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ACLGRAPH = ROOT / "vllm_ascend" / "worker" / "v2" / "spec_decode" / "autoregressive" / "aclgraph.py"
PATCH = ROOT / "vllm_ascend" / "patch" / "worker" / "patch_v2" / "patch_eagle_speculator.py"


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name} not found")


def test_eagle_aclgraph_uses_verified_main_speculator_contract() -> None:
    tree = ast.parse(ACLGRAPH.read_text())
    cls = _class(tree, "AutoRegressiveAclGraphManager")

    assert isinstance(cls.bases[0], ast.Name)
    assert cls.bases[0].id == "SpeculatorCudaGraphManager"

    source = ACLGRAPH.read_text()
    assert "AttentionStatePair" not in source
    assert "PrefillSpeculatorCudaGraphManager" not in source
    assert "DecodeSpeculatorCudaGraphManager" not in source


def test_draft_graph_passes_padded_shape_to_metadata_builder() -> None:
    tree = ast.parse(ACLGRAPH.read_text())
    run_fullgraph = _method(
        _class(tree, "AutoRegressiveAclGraphManager"),
        "run_fullgraph",
    )
    calls = [
        node
        for node in ast.walk(run_fullgraph)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "build_draft_attn_metadatas"
    ]

    assert len(calls) == 1
    assert [ast.unparse(arg) for arg in calls[0].args] == [
        "desc.num_reqs",
        "self.is_draft_model_prefill",
        "num_tokens",
    ]


def test_eagle_patch_replaces_verified_main_manager_symbol() -> None:
    source = PATCH.read_text()

    assert "vllm_speculator_module.SpeculatorCudaGraphManager = AutoRegressiveAclGraphManager" in source
    assert "PrefillSpeculatorCudaGraphManager" not in source
    assert "DecodeSpeculatorCudaGraphManager" not in source
