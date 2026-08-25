# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu import pcp_manager as vllm_pcp_manager
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pcp_manager import PCPManager

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2 import states as states_module
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.model_runner import (
    NPUModelRunner,
    _use_ascend_pcp_manager_for_vllm_0271,
)
from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager


def _mock_async_copy_to_cpu(value, out=None, device=None):
    """Copy PCP metadata without requiring device hooks in CPU-only UTs."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)

    if out is not None:
        out.copy_(value)
        return out

    return value.to(device="cpu")


def _make_local_pcp_batch():
    """Build a local batch in the shape returned by the community PCP manager."""
    input_buffers = AscendInputBuffers(
        max_num_reqs=4,
        max_num_tokens=16,
        device=torch.device("cpu"),
    )
    base_batch = InputBatch.make_dummy(
        num_reqs=2,
        num_tokens=6,
        input_buffers=input_buffers,
    )

    # Local PCP rows: one starts at position 6 and contains two tokens; the
    # other starts at position 13 and contains four tokens.
    base_batch.req_ids = ["req-head", "req-tail"]
    base_batch.idx_mapping = torch.tensor([3, 7], dtype=torch.int32)
    base_batch.idx_mapping_np = np.array([3, 7], dtype=np.int32)
    base_batch.expanded_idx_mapping = base_batch.idx_mapping
    base_batch.num_scheduled_tokens = np.array([2, 4], dtype=np.int32)
    base_batch.query_start_loc_np = np.array([0, 2, 6], dtype=np.int32)
    base_batch.query_start_loc.copy_(torch.tensor([0, 2, 6], dtype=torch.int32))
    base_batch.num_computed_tokens_np = np.array([6, 13], dtype=np.int32)
    base_batch.prefill_len_np = np.array([32, 32], dtype=np.int32)
    base_batch.num_computed_prefill_tokens_np = np.array([6, 13], dtype=np.int32)
    base_batch.is_prefilling_np = np.array([True, True])
    base_batch.seq_lens.copy_(torch.tensor([8, 17], dtype=torch.int32))
    base_batch.seq_lens_cpu_upper_bound = torch.tensor([500, 600], dtype=torch.int32)
    base_batch.input_ids.copy_(torch.tensor([10, 11, 20, 21, 22, 23], dtype=torch.int32))
    base_batch.positions.copy_(torch.tensor([6, 7, 13, 14, 15, 16], dtype=torch.int64))
    base_batch.is_padding.fill_(False)

    return AscendInputBatch(
        **base_batch.__dict__,
        seq_lens_np=np.array([101, 102], dtype=np.int32),
        attn_state="global-attn-state",
    )


def _make_global_pcp_batch():
    """Build the global batch that is passed into PCPManager.partition_batch."""
    input_buffers = AscendInputBuffers(
        max_num_reqs=4,
        max_num_tokens=32,
        device=torch.device("cpu"),
    )
    base_batch = InputBatch.make_dummy(
        num_reqs=1,
        num_tokens=18,
        input_buffers=input_buffers,
    )
    base_batch.req_ids = ["global-req"]
    base_batch.idx_mapping = torch.tensor([3], dtype=torch.int32)
    base_batch.idx_mapping_np = np.array([3], dtype=np.int32)
    base_batch.expanded_idx_mapping = base_batch.idx_mapping
    base_batch.num_scheduled_tokens = np.array([18], dtype=np.int32)
    base_batch.query_start_loc_np = np.array([0, 18], dtype=np.int32)
    base_batch.query_start_loc.copy_(torch.tensor([0, 18], dtype=torch.int32))
    base_batch.num_computed_tokens_np = np.array([0], dtype=np.int32)
    base_batch.prefill_len_np = np.array([18], dtype=np.int32)
    base_batch.num_computed_prefill_tokens_np = np.array([0], dtype=np.int32)
    base_batch.is_prefilling_np = np.array([True])
    base_batch.seq_lens.copy_(torch.tensor([18], dtype=torch.int32))
    base_batch.seq_lens_cpu_upper_bound = torch.tensor([18], dtype=torch.int32)
    base_batch.input_ids.copy_(torch.arange(18, dtype=torch.int32))
    base_batch.positions.copy_(torch.arange(18, dtype=torch.int64))
    base_batch.is_padding.fill_(False)

    return AscendInputBatch(
        **base_batch.__dict__,
        seq_lens_np=np.array([18], dtype=np.int32),
        attn_state="global-attn-state",
    )


def test_partition_batch_refreshes_local_ascend_input_batch_metadata():
    """Refresh Ascend metadata after the real PCP local-batch rewrite."""
    global_batch = _make_global_pcp_batch()
    req_states = SimpleNamespace(
        last_sampled_tokens=torch.zeros(4, dtype=torch.int64),
        prefill_len=SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int32)),
        draft_tokens=torch.empty((4, 0), dtype=torch.int64),
    )
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=req_states,
        max_num_reqs=1,
        max_num_tokens=18,
    )

    with (
        # This Triton helper is unrelated to PCP partitioning and has no CPU
        # implementation. Stub only it; AscendPCPManager.partition_batch and
        # PCPManager.partition_batch both execute unmocked below.
        patch(
            "vllm.v1.worker.gpu.pcp_manager.prepare_pos_seq_lens",
            return_value=None,
        ),
        patch(
            "vllm.v1.worker.gpu.pcp_manager.combine_sampled_and_draft_tokens",
            return_value=torch.zeros(2, dtype=torch.int64),
        ),
        patch(
            "vllm.v1.worker.gpu.pcp_manager.async_copy_to_gpu",
            side_effect=_mock_async_copy_to_cpu,
        ),
    ):
        result = manager.partition_batch(global_batch)

    assert isinstance(result, AscendInputBatch)
    assert result is not global_batch
    assert manager._global_batch is global_batch
    np.testing.assert_array_equal(global_batch.seq_lens_np, np.array([18], dtype=np.int32))
    assert global_batch.attn_state == "global-attn-state"

    # PCP=2 rank 0 owns the tail chunk then the head chunk; the real base
    # implementation produces this local row order and pads to rank 1's size.
    assert result.req_ids == ["global-req", "global-req"]
    np.testing.assert_array_equal(result.idx_mapping_np, np.array([3, 3], dtype=np.int32))
    np.testing.assert_array_equal(result.num_scheduled_tokens, np.array([3, 5], dtype=np.int32))
    np.testing.assert_array_equal(result.query_start_loc_np, np.array([0, 3, 8], dtype=np.int32))
    assert result.num_tokens == 8
    assert result.num_tokens_after_padding == 10
    assert torch.equal(result.input_ids[:8], torch.tensor([15, 16, 17, 0, 1, 2, 3, 4], dtype=torch.int32))

    # dataclasses.replace() retains the global Ascend-only fields by default;
    # the override must refresh them from real PCP-local CPU rows.
    expected_seq_lens = np.array([18, 5], dtype=np.int32)
    np.testing.assert_array_equal(result.seq_lens_np, expected_seq_lens)
    assert result.attn_state == "global-attn-state"


def test_dummy_attention_context_uses_rank_local_identity_view():
    manager = AscendPCPManager.__new__(AscendPCPManager)
    manager.pcp_world_size = 2
    manager.pcp_rank = 1
    manager.device = torch.device("cpu")
    input_batch = _make_local_pcp_batch()
    input_batch.is_dummy = True
    block_tables = (
        torch.tensor([[1]], dtype=torch.int32),
        torch.tensor([[2]], dtype=torch.int32),
    )
    slot_mappings = torch.arange(
        len(block_tables) * manager.pcp_world_size * input_batch.num_tokens,
        dtype=torch.int64,
    ).view(len(block_tables), -1)

    actual = manager.build_attention_context(
        input_batch,
        block_tables,
        slot_mappings,
    )

    expected_slot_mappings = slot_mappings.view(
        len(block_tables),
        manager.pcp_world_size,
        input_batch.num_tokens,
    )[:, manager.pcp_rank]
    restore_start = manager.pcp_rank * input_batch.num_tokens
    assert actual.global_batch is input_batch
    assert actual.global_block_tables is block_tables
    assert torch.equal(actual.global_slot_mappings, expected_slot_mappings)
    assert torch.equal(
        actual.hidden_restore_idx,
        torch.arange(
            restore_start,
            restore_start + input_batch.num_tokens,
        ),
    )
    assert actual.local_num_tokens_after_padding == input_batch.num_tokens


def test_prepare_slot_mappings_pads_each_pcp_rank_for_full_decode_graph() -> None:
    manager = AscendPCPManager.__new__(AscendPCPManager)
    manager.pcp_world_size = 2
    manager._global_batch = SimpleNamespace(
        num_tokens_after_padding=8,
        num_tokens=4,
        is_prefilling_np=np.array([False, False, False, False]),
    )
    manager._gathered_kv_slot_mappings = torch.full((1, 16), -99, dtype=torch.int64)
    compact_slot_mappings = manager._gathered_kv_slot_mappings[:, :8]
    compact_slot_mappings.copy_(torch.tensor([[10, 11, 12, 13, 20, 21, 22, 23]]))

    with patch.object(PCPManager, "prepare_slot_mappings", return_value=compact_slot_mappings):
        result = manager.prepare_slot_mappings()

    expected = torch.tensor([[10, 11, 12, 13, -1, -1, -1, -1, 20, 21, 22, 23, -1, -1, -1, -1]])
    assert torch.equal(result, expected)


def test_prepare_attn_uses_persistent_pad_filled_dummy_buffers() -> None:
    manager = AscendPCPManager.__new__(AscendPCPManager)
    manager.pcp_world_size = 2
    manager._local_block_tables = (
        torch.full((4, 3), 7, dtype=torch.int32),
    )
    manager._gathered_kv_slot_mappings = torch.zeros(
        (1, 16),
        dtype=torch.int64,
    )
    input_batch = SimpleNamespace(
        is_dummy=True,
        num_reqs_after_padding=2,
        num_tokens_after_padding=4,
    )

    block_tables, slot_mappings = manager.prepare_attn(input_batch)

    persistent_block_table = manager._local_block_tables[0]
    assert block_tables[0].data_ptr() == persistent_block_table.data_ptr()
    assert torch.equal(block_tables[0], torch.zeros(2, 3, dtype=torch.int32))
    assert torch.equal(
        persistent_block_table[2:],
        torch.full((2, 3), 7, dtype=torch.int32),
    )
    assert (
        slot_mappings.data_ptr()
        == manager._gathered_kv_slot_mappings.data_ptr()
    )
    assert slot_mappings.shape == (1, 8)
    assert torch.all(slot_mappings == -1)


def test_prepare_attn_delegates_runtime_batches() -> None:
    manager = AscendPCPManager.__new__(AscendPCPManager)
    input_batch = SimpleNamespace(is_dummy=False)
    expected = ((torch.empty(0),), torch.empty(0))

    with patch.object(
        PCPManager,
        "prepare_attn",
        return_value=expected,
    ) as prepare_attn:
        actual = manager.prepare_attn(input_batch)

    assert actual is expected
    prepare_attn.assert_called_once_with(input_batch)


def test_partition_batch_preserves_fia_dummy_layout() -> None:
    global_batch = _make_global_pcp_batch()
    global_batch.req_ids = ["decode-req"]
    global_batch.num_scheduled_tokens = np.array([1], dtype=np.int32)
    global_batch.num_tokens = 1
    global_batch.num_reqs_after_padding = 2
    global_batch.num_tokens_after_padding = 4
    global_batch.query_start_loc_np = np.array([0, 1, 4], dtype=np.int32)
    global_batch.query_start_loc = torch.tensor(
        [0, 1, 4],
        dtype=torch.int32,
    )
    global_batch.num_computed_tokens_np = np.array([10], dtype=np.int32)
    global_batch.prefill_len_np = np.array([10], dtype=np.int32)
    global_batch.num_computed_prefill_tokens_np = np.array(
        [10],
        dtype=np.int32,
    )
    global_batch.is_prefilling_np = np.array([False])
    global_batch.seq_lens = torch.tensor([11, 999], dtype=torch.int32)
    global_batch.seq_lens_cpu_upper_bound = torch.tensor(
        [11],
        dtype=torch.int32,
    )
    global_batch.input_ids[:4].copy_(
        torch.tensor([101, 999, 999, 999], dtype=torch.int32)
    )
    global_batch.positions[:4].copy_(
        torch.tensor([10, 999, 999, 999], dtype=torch.int64)
    )
    global_batch.is_padding[:4].fill_(False)

    req_states = SimpleNamespace(
        last_sampled_tokens=torch.zeros(2, dtype=torch.int64),
        prefill_len=SimpleNamespace(
            gpu=torch.zeros(2, dtype=torch.int32)
        ),
        draft_tokens=torch.empty((2, 0), dtype=torch.int64),
    )
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=req_states,
        max_num_reqs=2,
        max_num_tokens=4,
    )
    input_buffers = manager._input_buffers
    assert input_buffers is not None
    input_buffers.positions[0] = 10
    input_buffers.seq_lens[0] = 11

    with (
        patch(
            "vllm.v1.worker.gpu.pcp_manager.prepare_pos_seq_lens",
            return_value=None,
        ),
        patch(
            "vllm.v1.worker.gpu.pcp_manager.combine_sampled_and_draft_tokens",
            return_value=torch.zeros(1, dtype=torch.int64),
        ),
        patch(
            "vllm.v1.worker.gpu.pcp_manager.async_copy_to_gpu",
            side_effect=_mock_async_copy_to_cpu,
        ),
        patch(
            "vllm_ascend.worker.v2.pcp_manager.async_copy_to_gpu",
            side_effect=_mock_async_copy_to_cpu,
        ),
    ):
        local_batch = manager.partition_batch(global_batch)

    assert local_batch.num_reqs == 1
    assert local_batch.num_reqs_after_padding == 2
    assert local_batch.num_tokens == 1
    assert local_batch.num_tokens_after_padding == 4
    expected_query_start_loc = np.array([0, 1, 4], dtype=np.int32)
    np.testing.assert_array_equal(
        local_batch.query_start_loc_np,
        expected_query_start_loc,
    )
    torch.testing.assert_close(
        local_batch.query_start_loc,
        torch.from_numpy(expected_query_start_loc),
    )
    assert local_batch.input_ids.tolist() == [101, 0, 0, 0]
    assert local_batch.positions.tolist() == [10, 0, 0, 0]
    assert local_batch.seq_lens.tolist() == [11, 0]
    np.testing.assert_array_equal(
        local_batch.seq_lens_np,
        np.array([11, 0], dtype=np.int32),
    )
    assert local_batch.seq_lens_cpu_upper_bound.tolist() == [11, 0]
    assert local_batch.is_padding.tolist() == [False, True, True, True]
    assert manager._hidden_restore_idx is not None
    assert manager._hidden_restore_idx[1:4].tolist() == [0, 0, 0]


def test_partition_batch_restores_speculative_target_inputs() -> None:
    global_batch = _make_global_pcp_batch()
    global_batch.req_ids = ["spec-req"]
    global_batch.num_reqs_after_padding = 1
    global_batch.num_tokens = 2
    global_batch.num_tokens_after_padding = 2
    global_batch.input_ids = torch.tensor([101, 202], dtype=torch.int32)
    global_batch.positions = torch.tensor([10, 11], dtype=torch.int64)
    global_batch.is_padding = torch.zeros(2, dtype=torch.bool)
    global_batch.num_scheduled_tokens = np.array([2], dtype=np.int32)
    global_batch.num_computed_tokens_np = np.array([10], dtype=np.int32)
    global_batch.is_prefilling_np = np.array([False])
    global_batch.num_draft_tokens = 1
    global_batch.num_draft_tokens_per_req = np.array([1], dtype=np.int32)

    local_batch = replace(
        global_batch,
        input_ids=torch.zeros(2, dtype=torch.int32),
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        seq_lens_np=np.array([12], dtype=np.int32),
    )
    manager = AscendPCPManager.__new__(AscendPCPManager)
    manager.pcp_rank = 0
    manager.pcp_world_size = 2
    manager._padded_gather_idx = torch.tensor([0, 1, 0, 1])

    with patch.object(
        PCPManager,
        "partition_batch",
        return_value=local_batch,
    ) as parent_partition:
        result = manager.partition_batch(global_batch)

    parent_input = parent_partition.call_args.args[0]
    assert parent_input is not global_batch
    assert parent_input.num_draft_tokens == 0
    assert parent_input.num_draft_tokens_per_req is None
    assert manager._global_batch is global_batch
    assert result.input_ids.tolist() == [101, 202]
    np.testing.assert_array_equal(
        result.num_draft_tokens_per_req,
        np.array([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.seq_lens_np,
        np.array([12], dtype=np.int32),
    )


def test_request_state_cpu_and_numpy_tokens_share_storage() -> None:
    def init_base_state(
        state,
        max_num_reqs,
        max_model_len,
        max_num_batched_tokens,
        num_speculative_steps,
        vocab_size,
        device,
    ) -> None:
        state.max_num_reqs = max_num_reqs
        state.num_computed_tokens_np = np.zeros(max_num_reqs, dtype=np.int32)

    with patch.object(
        states_module.RequestState,
        "__init__",
        init_base_state,
    ):
        state = states_module.AscendRequestState(
            max_num_reqs=2,
            max_model_len=16,
            max_num_batched_tokens=16,
            num_speculative_steps=1,
            vocab_size=32,
            device=torch.device("cpu"),
        )

    assert state.num_computed_tokens_cpu.data_ptr() == state.num_computed_tokens_np.ctypes.data
    state.num_computed_tokens_np[0] = 17
    assert state.num_computed_tokens_cpu[0].item() == 17
    state.num_computed_tokens_cpu[1] = 23
    assert state.num_computed_tokens_np[1] == 23


def test_pcp_manager_restores_model_owned_hidden_buffer() -> None:
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, -1.0], [-1.0, -1.0]])
    restored = torch.tensor([[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]])
    manager = AscendPCPManager.__new__(AscendPCPManager)
    manager.pcp_world_size = 2
    manager._padded_gather_idx = torch.empty(6, dtype=torch.int64)

    captured_local_hidden_states = []

    def restore_hidden_states(value):
        captured_local_hidden_states.append(value.clone())
        return restored

    with patch.object(
        PCPManager,
        "restore_hidden_states",
        side_effect=restore_hidden_states,
    ) as restore_hidden_states_mock:
        manager.restore_hidden_state_buffer(hidden_states)

    restore_hidden_states_mock.assert_called_once()
    torch.testing.assert_close(
        captured_local_hidden_states[0],
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, -1.0]]),
    )
    torch.testing.assert_close(hidden_states[:3], restored)
    torch.testing.assert_close(hidden_states[3], torch.tensor([-1.0, -1.0]))


@pytest.mark.parametrize("method", ["mtp", "eagle3"])
def test_validate_config_allows_supported_speculators(method: str) -> None:
    speculative_config = SimpleNamespace(method=method)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2),
        speculative_config=speculative_config,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )

    with patch.object(PCPManager, "validate_config") as parent_validate:
        AscendPCPManager.validate_config(
            vllm_config,
            supports_mm_inputs=False,
        )

    validated_config = parent_validate.call_args.args[0]
    assert validated_config is not vllm_config
    assert validated_config.speculative_config is None
    assert vllm_config.speculative_config is speculative_config


def test_validate_config_rejects_unsupported_speculator() -> None:
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2),
        speculative_config=SimpleNamespace(method="ngram"),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )

    with (
        patch.object(PCPManager, "validate_config") as parent_validate,
        pytest.raises(
            NotImplementedError,
            match="only with MTP and Eagle3",
        ),
    ):
        AscendPCPManager.validate_config(
            vllm_config,
            supports_mm_inputs=False,
        )

    parent_validate.assert_not_called()


def test_mrv2_runner_registers_ascend_pcp_manager() -> None:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    assert runner.pcp_manager_cls is AscendPCPManager


@pytest.mark.skipif(
    not vllm_version_is("0.27.1"),
    reason="vLLM 0.27.1 is the only version with the hard-coded PCP manager builder",
)
def test_vllm_0271_builder_uses_ascend_pcp_manager() -> None:
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=16,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        ),
    )
    original_pcp_manager_cls = vllm_pcp_manager.PCPManager

    with (
        patch(
            "vllm.v1.worker.gpu.pcp_manager.get_pcp_group",
            return_value=SimpleNamespace(rank_in_group=0),
        ),
        patch.object(PCPManager, "validate_config") as upstream_validate_config,
        patch.object(AscendPCPManager, "__init__", return_value=None) as init_manager,
        _use_ascend_pcp_manager_for_vllm_0271(),
    ):
        manager = vllm_pcp_manager.maybe_build_pcp_manager(
            vllm_config,
            torch.device("cpu"),
            False,
            object(),
            object(),
        )

    assert type(manager) is AscendPCPManager
    assert vllm_pcp_manager.PCPManager is original_pcp_manager_cls
    upstream_validate_config.assert_called_once()
    validated_config, supports_mm_inputs = upstream_validate_config.call_args.args
    assert validated_config is not vllm_config
    assert validated_config.compilation_config is not vllm_config.compilation_config
    assert validated_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    assert supports_mm_inputs is False
    init_manager.assert_called_once()
