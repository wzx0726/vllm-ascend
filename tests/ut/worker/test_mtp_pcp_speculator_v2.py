# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.spec_decode.autoregressive import (
    speculator as speculator_module,
)
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
    AscendMTPSpeculator,
)


def _make_padded_input_batch() -> MagicMock:
    input_batch = MagicMock(spec=AscendInputBatch)
    input_batch.num_reqs = 2
    input_batch.num_reqs_after_padding = 4
    input_batch.num_tokens = 6
    input_batch.num_tokens_after_padding = 8
    input_batch.idx_mapping = torch.tensor([3, 7], dtype=torch.int32)
    input_batch.query_start_loc = torch.tensor([0, 3, 6, 6, 6], dtype=torch.int32)
    input_batch.query_start_loc_np = np.array([0, 3, 6, 6, 6], dtype=np.int32)
    input_batch.seq_lens = torch.arange(4, dtype=torch.int32)
    input_batch.seq_lens_cpu_upper_bound = torch.arange(4, dtype=torch.int32)
    input_batch.input_ids = torch.arange(8, dtype=torch.int32)
    input_batch.positions = torch.arange(8, dtype=torch.int64)
    input_batch.is_padding = torch.zeros(8, dtype=torch.bool)
    input_batch.seq_lens_np = np.arange(4, dtype=np.int32)
    return input_batch


@pytest.mark.parametrize(
    ("replicated_pcp", "expected_pcp_size"),
    [(True, 1), (False, 2)],
)
def test_draft_runtime_config_uses_draft_parallel_topology(
    replicated_pcp: bool,
    expected_pcp_size: int,
) -> None:
    draft_parallel_config = SimpleNamespace(
        prefill_context_parallel_size=2,
        enable_expert_parallel=False,
    )
    target_parallel_config = SimpleNamespace(rank=7)
    target_config = SimpleNamespace(
        parallel_config=target_parallel_config,
        speculative_config=SimpleNamespace(
            draft_parallel_config=draft_parallel_config,
        ),
    )
    draft_model_config = object()

    def fake_replace(config, **changes):
        values = vars(config).copy()
        values.update(changes)
        return SimpleNamespace(**values)

    with patch.object(
        speculator_module,
        "replace",
        side_effect=fake_replace,
    ):
        execution_config = AscendMTPSpeculator._create_draft_execution_config(target_config, replicated_pcp)
        speculator = object.__new__(AscendMTPSpeculator)
        speculator.vllm_config = execution_config
        speculator.draft_model_config = draft_model_config
        draft_config = speculator._create_draft_vllm_config()

    assert execution_config.parallel_config.rank == target_parallel_config.rank
    assert execution_config.parallel_config.prefill_context_parallel_size == expected_pcp_size
    assert draft_parallel_config.prefill_context_parallel_size == 2
    assert not execution_config.parallel_config.enable_expert_parallel
    assert draft_config.model_config is draft_model_config
    assert draft_config.parallel_config is execution_config.parallel_config


@pytest.mark.parametrize(("replicated_pcp", "manager_is_disabled"), [(True, True), (False, False)])
def test_draft_pcp_context_restores_manager_after_error(
    replicated_pcp: bool,
    manager_is_disabled: bool,
) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.pcp_manager = MagicMock()
    speculator.replicated_pcp = replicated_pcp
    speculator.model_state = SimpleNamespace(pcp_manager=speculator.pcp_manager)

    with pytest.raises(RuntimeError, match="proposal failed"), speculator._disable_target_pcp_for_replicated_draft():
        assert (speculator.model_state.pcp_manager is None) is manager_is_disabled
        raise RuntimeError("proposal failed")

    assert speculator.model_state.pcp_manager is speculator.pcp_manager


@pytest.mark.parametrize(
    ("replicated_pcp", "expected_source"),
    [(True, "draft"), (False, "target")],
)
def test_draft_prefill_attn_groups_follow_draft_topology(
    replicated_pcp: bool,
    expected_source: str,
) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = replicated_pcp
    speculator.attn_groups = [["draft"]]
    speculator.target_attn_groups = [["target"]]

    expected = speculator.attn_groups if expected_source == "draft" else speculator.target_attn_groups
    assert speculator.draft_prefill_attn_groups is expected


def test_prepare_replicated_prefill_attn_uses_global_batch() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.block_tables = MagicMock()
    speculator.kv_cache_config = object()
    speculator._build_draft_attn_metadata = MagicMock(return_value={"draft.layer": object()})
    input_batch = _make_padded_input_batch()
    global_slot_mapping = torch.arange(input_batch.num_tokens_after_padding).unsqueeze(0)
    slot_mappings = {"draft.layer": object()}
    speculator.block_tables.compute_slot_mappings.return_value = global_slot_mapping

    with patch.object(
        speculator_module,
        "build_slot_mappings_by_layer",
        return_value=slot_mappings,
    ) as build_slot_mappings:
        attn_metadata, actual_slot_mappings = speculator._prepare_replicated_prefill_attn(
            input_batch,
            input_batch.num_reqs_after_padding,
            input_batch.num_tokens_after_padding,
        )

    assert attn_metadata == speculator._build_draft_attn_metadata.return_value
    assert actual_slot_mappings is slot_mappings
    speculator.block_tables.gather_block_tables.assert_called_once_with(
        input_batch.idx_mapping,
        num_reqs_padded=input_batch.num_reqs_after_padding,
    )
    speculator.block_tables.compute_slot_mappings.assert_called_once_with(
        input_batch.idx_mapping,
        input_batch.query_start_loc,
        input_batch.positions,
        num_tokens_padded=input_batch.num_tokens_after_padding,
    )
    build_slot_mappings.assert_called_once_with(
        global_slot_mapping,
        speculator.kv_cache_config,
    )
    speculator._build_draft_attn_metadata.assert_called_once_with(
        num_reqs=input_batch.num_reqs,
        num_reqs_padded=input_batch.num_reqs_after_padding,
        num_tokens_padded=input_batch.num_tokens_after_padding,
        seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
        step=0,
        query_start_loc_np=input_batch.query_start_loc_np,
    )


def test_prefill_rebuilds_replicated_pcp_metadata_before_filtering() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = True
    speculator.input_batch = _make_padded_input_batch()
    speculator.input_batch.is_dummy = False
    speculator.draft_attn_layer_names = {"draft.layer"}

    draft_metadata = object()
    global_slot_mappings = MagicMock()
    speculator._prepare_replicated_prefill_attn = MagicMock(
        return_value=(
            {
                "draft.layer": draft_metadata,
                "target.layer": object(),
            },
            global_slot_mappings,
        )
    )

    with patch.object(
        speculator_module.AutoRegressiveSpeculator,
        "_prefill",
    ) as parent_prefill:
        speculator._prefill(
            num_reqs=2,
            num_tokens=8,
            attn_metadata={"local.layer": object()},
            slot_mappings=MagicMock(),
            num_tokens_across_dp=None,
        )

    speculator._prepare_replicated_prefill_attn.assert_called_once_with(
        speculator.input_batch,
        2,
        8,
    )
    parent_prefill.assert_called_once()
    parent_args = parent_prefill.call_args.args
    assert parent_args[:2] == (2, 8)
    assert parent_args[2] == {"draft.layer": draft_metadata}
    assert parent_args[3] is global_slot_mappings


def test_graph_prefill_rebuilds_replicated_pcp_metadata() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = True
    speculator.input_batch = _make_padded_input_batch()
    speculator.draft_attn_layer_names = {"draft.layer"}
    draft_metadata = object()
    speculator._prepare_replicated_prefill_attn = MagicMock(
        return_value=(
            {
                "draft.layer": draft_metadata,
                "target.layer": object(),
            },
            MagicMock(),
        )
    )

    actual = speculator.build_draft_attn_metadatas(
        num_reqs_padded=4,
        is_draft_model_prefill=True,
        num_tokens_padded=8,
    )

    assert actual == [{"draft.layer": draft_metadata}]
    speculator._prepare_replicated_prefill_attn.assert_called_once_with(
        speculator.input_batch,
        4,
        8,
    )


def test_propose_disables_target_pcp_manager_for_replicated_draft() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = True
    speculator.input_batch = None
    speculator.pcp_manager = MagicMock()
    speculator.model_state = SimpleNamespace(
        pcp_manager=speculator.pcp_manager,
    )
    input_batch = _make_padded_input_batch()
    expected = object()

    def parent_propose(*args, **kwargs):
        assert args[0] is input_batch
        assert speculator.model_state.pcp_manager is None
        return expected

    with (
        patch.object(
            MTPSpeculator,
            "propose",
            side_effect=parent_propose,
        ),
        patch.object(
            speculator_module,
            "build_attn_metadata_wrapper",
            return_value=nullcontext(),
        ),
        patch.object(
            speculator_module,
            "torch_gather_wrapper",
            return_value=nullcontext(),
        ),
    ):
        actual = speculator.propose(
            input_batch,
            *[MagicMock() for _ in range(10)],
        )

    assert actual is expected
    assert speculator.input_batch is input_batch
    assert speculator.model_state.pcp_manager is speculator.pcp_manager
