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
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pcp_manager import PCPManager

import vllm_ascend.worker.v2.pcp_manager as pcp_manager_module
import vllm_ascend.worker.v2.states as states_module
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
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


def _make_local_pcp_batch() -> AscendInputBatch:
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


def _make_global_spec_decode_batch(
    num_draft_tokens: int = 1,
) -> AscendInputBatch:
    batch = _make_global_pcp_batch()
    num_tokens = num_draft_tokens + 1
    batch.req_ids = ["spec-req"]
    batch.num_scheduled_tokens = np.array([num_tokens], dtype=np.int32)
    batch.num_tokens = num_tokens
    batch.num_tokens_after_padding = num_tokens
    batch.query_start_loc_np = np.array([0, num_tokens], dtype=np.int32)
    batch.query_start_loc[:2].copy_(torch.tensor([0, num_tokens], dtype=torch.int32))
    batch.num_computed_tokens_np = np.array([10], dtype=np.int32)
    batch.prefill_len_np = np.array([10], dtype=np.int32)
    batch.num_computed_prefill_tokens_np = np.array([10], dtype=np.int32)
    batch.is_prefilling_np = np.array([False])
    batch.seq_lens[:1].copy_(torch.tensor([10 + num_tokens], dtype=torch.int32))
    batch.seq_lens_cpu_upper_bound = torch.tensor(
        [10 + num_tokens],
        dtype=torch.int32,
    )
    batch.input_ids[0] = 101
    batch.input_ids[1:num_tokens].copy_(torch.arange(202, 202 + num_draft_tokens, dtype=torch.int32))
    batch.cu_num_logits_np = np.array([0, num_tokens], dtype=np.int32)
    batch.cu_num_logits[:2].copy_(torch.tensor([0, num_tokens], dtype=torch.int32))
    batch.expanded_idx_mapping = torch.full((num_tokens,), 3, dtype=torch.int32)
    batch.expanded_local_pos = torch.arange(num_tokens, dtype=torch.int32)
    batch.logits_indices = torch.arange(num_tokens, dtype=torch.int64)
    batch.positions[:num_tokens].copy_(torch.arange(10, 10 + num_tokens, dtype=torch.int64))
    batch.is_padding[:num_tokens].fill_(False)
    batch.num_draft_tokens = num_draft_tokens
    batch.num_draft_tokens_per_req = np.array(
        [num_draft_tokens],
        dtype=np.int32,
    )
    batch.seq_lens_np = np.array([10 + num_tokens], dtype=np.int32)
    batch.attn_state = AscendAttentionState.SpecDecoding
    return batch


def test_mtp_rejection_syncs_corrected_num_computed_tokens_to_numpy() -> None:
    num_computed_tokens_np = np.array([0, 309], dtype=np.int32)
    runner = SimpleNamespace(
        speculator=object(),
        num_computed_tokens_event=MagicMock(),
        num_computed_tokens_cpu=torch.tensor([0, 308], dtype=torch.int32),
        req_states=SimpleNamespace(
            req_id_to_index={"req": 1},
            num_computed_tokens_cpu=torch.from_numpy(num_computed_tokens_np),
            num_computed_tokens_np=num_computed_tokens_np,
        ),
        input_buffers=SimpleNamespace(
            seq_lens_cpu=torch.zeros(1, dtype=torch.int32),
        ),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 2},
        scheduled_cached_reqs=SimpleNamespace(req_ids=["req"]),
    )

    NPUModelRunner._update_seq_lens_cpu(
        runner,
        scheduler_output,
        req_ids=["req"],
    )

    runner.num_computed_tokens_event.synchronize.assert_called_once_with()
    assert runner.req_states.num_computed_tokens_cpu[1].item() == 308
    assert runner.req_states.num_computed_tokens_np[1] == 308
    assert runner.input_buffers.seq_lens_cpu[0].item() == 310


def test_request_state_num_computed_tokens_cpu_shares_numpy_storage() -> None:
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


def test_pcp_manager_restores_model_specific_hidden_state_buffer() -> None:
    mtp_target_hidden_states = torch.full((4, 2), -1.0)
    local_hidden_states = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    mtp_target_hidden_states[:2].copy_(local_hidden_states)
    restored_hidden_states = torch.tensor(
        [
            [1.0, 2.0],
            [5.0, 6.0],
            [3.0, 4.0],
        ]
    )
    manager = AscendPCPManager.__new__(AscendPCPManager)
    manager.pcp_world_size = 2
    manager._padded_gather_idx = torch.empty(6, dtype=torch.int64)
    captured_local_hidden_states = []

    def restore_hidden_states(hidden_states):
        captured_local_hidden_states.append(hidden_states.clone())
        return restored_hidden_states

    with patch.object(
        PCPManager,
        "restore_hidden_states",
        side_effect=restore_hidden_states,
    ) as restore_hidden_states:
        result = manager.restore_hidden_state_buffer(mtp_target_hidden_states)

    assert result is None
    restore_hidden_states.assert_called_once()
    torch.testing.assert_close(
        captured_local_hidden_states[0],
        torch.cat([local_hidden_states, torch.full((1, 2), -1.0)]),
    )
    torch.testing.assert_close(
        mtp_target_hidden_states[:3],
        restored_hidden_states,
    )
    torch.testing.assert_close(
        mtp_target_hidden_states[3],
        torch.tensor([-1.0, -1.0]),
    )


def test_partition_batch_refreshes_local_ascend_input_batch_metadata():
    """Refresh Ascend metadata after the real PCP local-batch rewrite."""
    vllm_config = object()
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
        vllm_config=vllm_config,
        req_states=req_states,
        max_num_reqs=1,
        max_num_tokens=18,
    )
    attn_state = MagicMock()

    with (
        # This Triton helper is unrelated to PCP partitioning and has no CPU
        # implementation. Stub only it; the Ascend partition override
        # executes unmocked below.
        patch(
            "vllm_ascend.worker.v2.pcp_manager.prepare_pos_seq_lens",
            return_value=None,
        ),
        patch(
            "vllm_ascend.worker.v2.pcp_manager.async_copy_to_gpu",
            side_effect=_mock_async_copy_to_cpu,
        ),
        patch.object(
            pcp_manager_module,
            "build_attn_state",
            return_value=attn_state,
        ) as build_attn_state,
    ):
        result = manager.partition_batch(global_batch)

    assert isinstance(result, AscendInputBatch)
    assert result is not global_batch
    assert manager._global_batch is global_batch
    np.testing.assert_array_equal(
        global_batch.seq_lens_np,
        np.array([18], dtype=np.int32),
    )
    assert global_batch.attn_state == "global-attn-state"

    # PCP=2 rank 0 owns the tail chunk then the head chunk; the real base
    # implementation produces this local row order and pads to rank 1's size.
    assert result.req_ids == ["global-req", "global-req"]
    np.testing.assert_array_equal(
        result.idx_mapping_np,
        np.array([3, 3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.num_scheduled_tokens,
        np.array([3, 5], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.query_start_loc_np,
        np.array([0, 3, 8], dtype=np.int32),
    )
    assert result.num_tokens == 8
    assert result.num_tokens_after_padding == 10
    assert torch.equal(
        result.input_ids[:8],
        torch.tensor(
            [15, 16, 17, 0, 1, 2, 3, 4],
            dtype=torch.int32,
        ),
    )

    # dataclasses.replace() retains the global Ascend-only fields by default;
    # the override must refresh them from real PCP-local CPU rows.
    expected_seq_lens = np.array([18, 5], dtype=np.int32)
    np.testing.assert_array_equal(result.seq_lens_np, expected_seq_lens)
    assert result.attn_state is attn_state

    args = build_attn_state.call_args.args
    assert args[0] is vllm_config
    np.testing.assert_array_equal(args[1], expected_seq_lens)
    assert args[2] == 2
    np.testing.assert_array_equal(args[3], np.array([3, 5], dtype=np.int32))
    np.testing.assert_array_equal(args[4], np.array([3, 5], dtype=np.int32))


@pytest.mark.parametrize("method", ["mtp", "eagle3"])
def test_partition_batch_preserves_global_speculative_metadata(
    method: str,
) -> None:
    global_batch = _make_global_spec_decode_batch()
    global_batch.num_reqs_after_padding = 2
    global_batch.num_tokens_after_padding = 4
    global_batch.query_start_loc_np = np.array([0, 2, 4], dtype=np.int32)
    global_batch.query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=MagicMock(),
        max_num_reqs=2,
        max_num_tokens=18,
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(method=method),
            num_speculative_tokens=1,
        ),
    )

    with (
        patch.object(
            PCPManager,
            "partition_batch",
            side_effect=AssertionError("Ascend must not call upstream partition"),
        ),
        patch.object(
            pcp_manager_module,
            "prepare_pos_seq_lens",
            return_value=None,
        ),
        patch.object(
            pcp_manager_module,
            "async_copy_to_gpu",
            side_effect=_mock_async_copy_to_cpu,
        ),
    ):
        local_batch = manager.partition_batch(global_batch)

    assert manager._global_batch is global_batch
    assert manager._hidden_restore_idx is not None
    assert manager._hidden_restore_idx.tolist() == [0, 1, 0, 0]

    local_hidden_states = torch.tensor([[10.0], [11.0], [99.0], [99.0]])
    gathered_hidden_states = torch.tensor([[10.0], [11.0], [99.0], [99.0], [20.0], [21.0], [99.0], [99.0]])
    pcp_group = MagicMock()
    pcp_group.all_gather.return_value = gathered_hidden_states
    with patch(
        "vllm.v1.worker.gpu.pcp_manager.get_pcp_group",
        return_value=pcp_group,
    ):
        restored_hidden_states = manager.restore_hidden_states(local_hidden_states)
    pcp_group.all_gather.assert_called_once_with(local_hidden_states, dim=0)
    torch.testing.assert_close(
        restored_hidden_states,
        torch.tensor([[10.0], [11.0], [10.0], [10.0]]),
    )

    assert global_batch.num_draft_tokens == 1
    np.testing.assert_array_equal(
        global_batch.num_draft_tokens_per_req,
        np.array([1], dtype=np.int32),
    )
    assert global_batch.input_ids[:2].tolist() == [101, 202]
    np.testing.assert_array_equal(
        global_batch.cu_num_logits_np,
        np.array([0, 2], dtype=np.int32),
    )
    assert global_batch.cu_num_logits.tolist() == [0, 2]
    assert global_batch.expanded_idx_mapping.tolist() == [3, 3]
    assert global_batch.expanded_local_pos.tolist() == [0, 1]
    assert global_batch.logits_indices.tolist() == [0, 1]
    assert local_batch.req_ids == ["spec-req"]
    assert local_batch.input_ids[:2].tolist() == [101, 202]
    assert local_batch.num_draft_tokens == 0
    assert local_batch.num_draft_tokens_per_req is None
    np.testing.assert_array_equal(
        local_batch.cu_num_logits_np,
        np.array([0, 1], dtype=np.int32),
    )
    assert local_batch.cu_num_logits.tolist() == [0, 1]
    assert local_batch.expanded_idx_mapping.tolist() == [3]
    assert local_batch.expanded_local_pos.tolist() == [0]
    assert local_batch.logits_indices.tolist() == [1]


@pytest.mark.parametrize("method", ["mtp", "eagle3"])
def test_mixed_spec_decode_batch_builds_attention_from_valid_token_counts(
    method: str,
) -> None:
    global_batch = _make_local_pcp_batch()
    global_batch.num_scheduled_tokens = np.array([1, 2], dtype=np.int32)
    global_batch.num_tokens = 3
    global_batch.num_tokens_after_padding = 3
    global_batch.query_start_loc_np = np.array([0, 1, 3], dtype=np.int32)
    global_batch.query_start_loc[:3].copy_(torch.tensor([0, 1, 3], dtype=torch.int32))
    global_batch.num_computed_tokens_np = np.array([0, 10], dtype=np.int32)
    global_batch.prefill_len_np = np.array([5, 10], dtype=np.int32)
    global_batch.num_computed_prefill_tokens_np = np.array(
        [0, 10],
        dtype=np.int32,
    )
    global_batch.is_prefilling_np = np.array([True, False])
    global_batch.seq_lens[:2].copy_(torch.tensor([1, 12], dtype=torch.int32))
    global_batch.seq_lens_cpu_upper_bound = torch.tensor(
        [1, 12],
        dtype=torch.int32,
    )
    global_batch.input_ids[:3].copy_(torch.tensor([10, 20, 21], dtype=torch.int32))
    global_batch.positions[:3].copy_(torch.tensor([0, 10, 11], dtype=torch.int64))
    global_batch.num_draft_tokens = 1
    global_batch.num_draft_tokens_per_req = np.array(
        [0, 1],
        dtype=np.int32,
    )

    req_states = SimpleNamespace(
        last_sampled_tokens=torch.zeros(8, dtype=torch.int64),
        prefill_len=SimpleNamespace(gpu=torch.zeros(8, dtype=torch.int32)),
        draft_tokens=torch.zeros((8, 1), dtype=torch.int64),
    )
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=req_states,
        max_num_reqs=2,
        max_num_tokens=6,
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(method=method),
            num_speculative_tokens=1,
        ),
    )
    attn_state = MagicMock()

    with (
        patch.object(
            pcp_manager_module,
            "prepare_pos_seq_lens",
            return_value=None,
        ),
        patch.object(
            pcp_manager_module,
            "async_copy_to_gpu",
            side_effect=_mock_async_copy_to_cpu,
        ),
        patch.object(
            pcp_manager_module,
            "build_attn_state",
            return_value=attn_state,
        ) as build_attn_state,
    ):
        local_batch = manager.partition_batch(global_batch)

    assert manager._global_batch is global_batch
    assert local_batch.req_ids == ["req-tail", "req-head"]
    np.testing.assert_array_equal(
        local_batch.num_scheduled_tokens,
        np.array([2, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        global_batch.num_draft_tokens_per_req,
        np.array([0, 1], dtype=np.int32),
    )
    assert local_batch.input_ids[:3].tolist() == [20, 21, 10]
    assert local_batch.num_draft_tokens_per_req is None
    assert local_batch.num_draft_tokens == 0
    np.testing.assert_array_equal(
        local_batch.cu_num_logits_np,
        np.array([0, 1, 2], dtype=np.int32),
    )
    assert local_batch.expanded_idx_mapping.tolist() == [7, 3]

    args = build_attn_state.call_args.args
    assert local_batch.expanded_local_pos.tolist() == [0, 0]
    assert local_batch.logits_indices.tolist() == [1, 2]
    np.testing.assert_array_equal(
        args[3],
        np.array([2, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        args[4],
        np.array([1, 1], dtype=np.int32),
    )


@pytest.mark.parametrize(
    ("method", "expected_state"),
    [
        ("mtp", AscendAttentionState.SpecDecoding),
        ("eagle3", AscendAttentionState.ChunkedPrefill),
    ],
)
def test_spec_decode_attention_state_remains_method_specific(
    method: str,
    expected_state: AscendAttentionState,
) -> None:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(runner_type="generate"),
        speculative_config=SimpleNamespace(method=method),
    )

    state = pcp_manager_module.build_attn_state(
        vllm_config,
        seq_lens_np=np.array([12], dtype=np.int32),
        num_reqs=1,
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        num_valid_tokens=np.array([1], dtype=np.int32),
    )

    assert state is expected_state


def test_partition_batch_requires_per_request_draft_counts() -> None:
    global_batch = _make_global_spec_decode_batch()
    global_batch.num_draft_tokens_per_req = None
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=MagicMock(),
        max_num_reqs=1,
        max_num_tokens=18,
        vllm_config=SimpleNamespace(num_speculative_tokens=1),
    )

    with pytest.raises(
        RuntimeError,
        match="requires per-request draft token counts",
    ):
        manager.partition_batch(global_batch)

    assert manager._global_batch is global_batch


def test_partition_batch_rejects_draft_tokens_on_prefill_requests() -> None:
    global_batch = _make_global_pcp_batch()
    global_batch.num_draft_tokens = 1
    global_batch.num_draft_tokens_per_req = np.array([1], dtype=np.int32)
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=MagicMock(),
        max_num_reqs=1,
        max_num_tokens=18,
        vllm_config=SimpleNamespace(num_speculative_tokens=1),
    )

    with pytest.raises(
        RuntimeError,
        match="does not support draft tokens on prefill requests",
    ):
        manager.partition_batch(global_batch)

    assert manager._global_batch is global_batch


def test_prepare_slot_mappings_extends_decode_layout_for_graph_padding() -> None:
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=MagicMock(),
        max_num_reqs=2,
        max_num_tokens=4,
        vllm_config=object(),
    )
    global_batch = _make_decode_local_batch(manager, num_reqs=2)
    global_batch.is_prefilling_np = np.array([False, False])
    manager._global_batch = replace(
        global_batch,
        num_tokens_after_padding=4,
    )
    manager._gathered_kv_slot_mappings = torch.full(
        (1, 8),
        999,
        dtype=torch.int64,
    )
    base_slot_mappings = manager._gathered_kv_slot_mappings[:, :4]
    base_slot_mappings.copy_(torch.tensor([[3415, 3416, 3415, 3416]]))

    with patch.object(
        PCPManager,
        "prepare_slot_mappings",
        return_value=base_slot_mappings,
    ) as prepare_slot_mappings:
        result = manager.prepare_slot_mappings()

    prepare_slot_mappings.assert_called_once_with()
    assert result.tolist() == [[3415, 3416, 3415, 3416] + [PAD_SLOT_ID] * 4]


def test_prepare_slot_mappings_keeps_prefill_layout() -> None:
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=MagicMock(),
        max_num_reqs=2,
        max_num_tokens=4,
        vllm_config=object(),
    )
    global_batch = _make_decode_local_batch(manager, num_reqs=2)
    manager._global_batch = replace(
        global_batch,
        is_prefilling_np=np.array([True, False]),
        num_tokens_after_padding=4,
    )
    manager._gathered_kv_slot_mappings = torch.empty(
        (1, 8),
        dtype=torch.int64,
    )
    base_slot_mappings = torch.tensor([[41, 42]], dtype=torch.int64)

    with patch.object(
        PCPManager,
        "prepare_slot_mappings",
        return_value=base_slot_mappings,
    ) as prepare_slot_mappings:
        result = manager.prepare_slot_mappings()

    assert result is base_slot_mappings
    prepare_slot_mappings.assert_called_once_with()


def test_cached_prefill_partitions_only_the_scheduled_suffix() -> None:
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=MagicMock(),
        max_num_reqs=4,
        max_num_tokens=8,
        vllm_config=object(),
    )

    def local_starts(num_computed_tokens: int) -> list[int]:
        segments = manager._get_rank_segments(
            rank=0,
            num_scheduled_tokens=np.array([8], dtype=np.int32),
            num_computed_tokens=np.array([num_computed_tokens], dtype=np.int32),
            is_prefilling=np.array([True]),
            query_start_loc_np=np.array([0, 8], dtype=np.int32),
        )
        return [num_computed_tokens + segment.global_batch_slice.start for segment in segments]

    # Two scheduler iterations of one longer suffix advance from the cached
    # prefix without repartitioning or recomputing that prefix.
    assert local_starts(128) == [128, 134]
    assert local_starts(136) == [136, 142]


def test_pcp_layout_orders_cache_hit_miss_and_decode_rows() -> None:
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=MagicMock(),
        max_num_reqs=8,
        max_num_tokens=8,
        vllm_config=object(),
    )
    num_scheduled_tokens = np.array([1, 4, 1], dtype=np.int32)
    num_computed_tokens = np.array([128, 0, 256], dtype=np.int32)
    is_prefilling = np.array([True, True, False])
    query_start_loc = np.array([0, 1, 5, 6], dtype=np.int32)

    rank_zero = manager._get_rank_segments(
        0,
        num_scheduled_tokens,
        num_computed_tokens,
        is_prefilling,
        query_start_loc,
    )
    rank_one = manager._get_rank_segments(
        1,
        num_scheduled_tokens,
        num_computed_tokens,
        is_prefilling,
        query_start_loc,
    )

    def layout(segments):
        return [
            (
                segment.global_batch_req_idx,
                segment.global_batch_slice.start,
                segment.global_batch_slice.stop,
            )
            for segment in segments
        ]

    # Continued prefills and replicated decodes stay before fresh prefills.
    # The one-token cache-hit suffix is owned by rank 0; no cached prefix token
    # appears in either rank's scheduled slices.
    assert layout(rank_zero) == [
        (0, 0, 1),
        (2, 5, 6),
        (1, 1, 2),
        (1, 4, 5),
    ]
    assert layout(rank_one) == [
        (2, 5, 6),
        (1, 2, 3),
        (1, 3, 4),
    ]


def test_npu_model_runner_uses_ascend_pcp_manager() -> None:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    assert runner.pcp_manager_cls is AscendPCPManager


def test_initialize_kv_cache_skips_pcp_binding_when_disabled() -> None:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.pcp_manager = None
    runner.model_config = MagicMock(enable_return_routed_experts=False)
    kv_cache_config = MagicMock()

    with (
        patch("vllm_ascend.worker.v2.model_runner.graph_manager_wrapper"),
        patch("vllm.v1.worker.gpu.model_runner.GPUModelRunner.initialize_kv_cache") as initialize_kv_cache,
    ):
        runner.initialize_kv_cache(kv_cache_config)

    initialize_kv_cache.assert_called_once_with(kv_cache_config)


@pytest.mark.parametrize("method", ["mtp", "eagle3"])
def test_validate_config_accepts_supported_speculative_methods(
    method: str,
) -> None:
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(),
        ),
        speculative_config=SimpleNamespace(method=method),
        lora_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
        ),
    )

    AscendPCPManager.validate_config(
        vllm_config,
        supports_mm_inputs=False,
    )


def test_validate_config_rejects_unsupported_speculative_method() -> None:
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(),
        ),
        speculative_config=SimpleNamespace(method="ngram"),
        lora_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
        ),
    )

    with pytest.raises(
        NotImplementedError,
        match="only with MTP and Eagle3",
    ):
        AscendPCPManager.validate_config(
            vllm_config,
            supports_mm_inputs=False,
        )


def _make_decode_local_batch(
    manager: AscendPCPManager,
    num_reqs: int,
) -> AscendInputBatch:
    input_buffers = manager._input_buffers
    assert input_buffers is not None
    base_batch = InputBatch.make_dummy(
        num_reqs=num_reqs,
        num_tokens=num_reqs,
        input_buffers=input_buffers,
    )
    base_batch.num_computed_tokens_np = np.arange(
        10,
        10 + num_reqs,
        dtype=np.int32,
    )
    base_batch.num_scheduled_tokens = np.ones(num_reqs, dtype=np.int32)
    seq_lens = base_batch.num_computed_tokens_np + base_batch.num_scheduled_tokens
    base_batch.seq_lens.copy_(torch.from_numpy(seq_lens))
    base_batch.seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens)
    base_batch.input_ids.copy_(torch.arange(100, 100 + num_reqs, dtype=torch.int32))
    base_batch.positions.copy_(torch.arange(10, 10 + num_reqs, dtype=torch.int64))
    base_batch.is_padding.fill_(False)
    return AscendInputBatch(
        **base_batch.__dict__,
        seq_lens_np=seq_lens,
        attn_state="decode-attn-state",
    )


def test_pad_decode_batch_supports_single_fia_dummy_request():
    """Allow one FIA dummy request to consume multiple padding tokens."""
    num_reqs = 2
    graph_num_reqs = 3
    graph_num_tokens = 4
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=MagicMock(),
        max_num_reqs=graph_num_reqs,
        max_num_tokens=graph_num_tokens,
        vllm_config=object(),
    )
    local_batch = _make_decode_local_batch(manager, num_reqs=num_reqs)
    input_buffers = manager._input_buffers
    assert input_buffers is not None
    input_buffers.input_ids[num_reqs:graph_num_tokens].fill_(999)
    input_buffers.positions[num_reqs:graph_num_tokens].fill_(999)
    input_buffers.seq_lens[num_reqs:graph_num_reqs].fill_(999)
    input_buffers.is_padding[num_reqs:graph_num_tokens].fill_(False)

    graph_input_ids = torch.full(
        (graph_num_tokens,),
        999,
        dtype=torch.int32,
    )
    graph_input_ids[:num_reqs].copy_(local_batch.input_ids)
    graph_positions = torch.full(
        (graph_num_tokens,),
        999,
        dtype=torch.int64,
    )
    graph_positions[:num_reqs].copy_(local_batch.positions)
    manager._global_batch = replace(
        local_batch,
        num_reqs_after_padding=graph_num_reqs,
        num_tokens_after_padding=graph_num_tokens,
        input_ids=graph_input_ids,
        positions=graph_positions,
        is_padding=torch.ones(graph_num_tokens, dtype=torch.bool),
    )

    result = manager._pad_decode_batch_for_full_graph(
        local_batch,
        graph_num_reqs,
        graph_num_tokens,
        local_batch.seq_lens_np,
    )

    expected_query_start_loc = np.array([0, 1, 2, 4], dtype=np.int32)
    np.testing.assert_array_equal(
        result.query_start_loc_np,
        expected_query_start_loc,
    )
    torch.testing.assert_close(
        result.query_start_loc,
        torch.from_numpy(expected_query_start_loc),
    )
    assert result.input_ids.tolist() == [100, 101, 0, 0]
    assert result.positions.tolist() == [10, 11, 0, 0]
    assert result.seq_lens.tolist() == [11, 12, 0]
    assert result.is_padding.tolist() == [False, False, True, True]


def test_pad_decode_batch_uses_k_plus_one_tokens_per_dummy_request() -> None:
    num_reqs = 2
    query_len = 4
    graph_num_reqs = 3
    graph_num_tokens = graph_num_reqs * query_len
    manager = AscendPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        req_states=MagicMock(),
        max_num_reqs=graph_num_reqs,
        max_num_tokens=graph_num_tokens,
        vllm_config=SimpleNamespace(num_speculative_tokens=3),
    )
    local_batch = _make_decode_local_batch(manager, num_reqs=num_reqs)
    input_buffers = manager._input_buffers
    assert input_buffers is not None

    num_tokens = num_reqs * query_len
    query_start_loc_np = np.array([0, 4, 8], dtype=np.int32)
    input_buffers.query_start_loc[: num_reqs + 1].copy_(torch.from_numpy(query_start_loc_np))
    input_buffers.input_ids[:num_tokens].copy_(torch.arange(100, 100 + num_tokens, dtype=torch.int32))
    input_buffers.positions[:num_tokens].copy_(torch.arange(10, 10 + num_tokens, dtype=torch.int64))
    input_buffers.is_padding[:num_tokens].fill_(False)
    local_seq_lens_np = np.array([14, 15], dtype=np.int32)
    input_buffers.seq_lens[:num_reqs].copy_(torch.from_numpy(local_seq_lens_np))
    local_batch = replace(
        local_batch,
        num_scheduled_tokens=np.full(num_reqs, query_len, dtype=np.int32),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        query_start_loc=input_buffers.query_start_loc[: num_reqs + 1],
        query_start_loc_np=query_start_loc_np,
        seq_lens=input_buffers.seq_lens[:num_reqs],
        seq_lens_cpu_upper_bound=torch.from_numpy(local_seq_lens_np.copy()),
        input_ids=input_buffers.input_ids[:num_tokens],
        positions=input_buffers.positions[:num_tokens],
        is_padding=input_buffers.is_padding[:num_tokens],
        seq_lens_np=local_seq_lens_np,
    )
    manager._global_batch = replace(
        local_batch,
        num_reqs_after_padding=graph_num_reqs,
        num_tokens_after_padding=graph_num_tokens,
        input_ids=torch.full((graph_num_tokens,), 999, dtype=torch.int32),
        positions=torch.full((graph_num_tokens,), 999, dtype=torch.int64),
        is_padding=torch.ones(graph_num_tokens, dtype=torch.bool),
    )

    result = manager._pad_decode_batch_for_full_graph(
        local_batch,
        graph_num_reqs,
        graph_num_tokens,
        local_seq_lens_np,
    )

    expected_query_start_loc = np.array([0, 4, 8, 12], dtype=np.int32)
    np.testing.assert_array_equal(
        result.query_start_loc_np,
        expected_query_start_loc,
    )
    torch.testing.assert_close(
        result.query_start_loc,
        torch.from_numpy(expected_query_start_loc),
    )
    assert result.seq_lens.tolist() == [14, 15, 0]
    assert result.is_padding.tolist() == [False] * 8 + [True] * 4
