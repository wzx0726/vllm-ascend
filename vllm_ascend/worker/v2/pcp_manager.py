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

from copy import copy
from dataclasses import dataclass, replace

import numpy as np
import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.pcp_manager import PCPManager

from vllm_ascend.worker.v2.input_batch import AscendInputBatch


@dataclass(frozen=True)
class AscendPCPAttentionContext:
    """Canonical global PCP view for one attention step."""

    # The global batch and its associated metadata, used to build DSA attention metadata.
    global_batch: AscendInputBatch
    global_block_tables: tuple[torch.Tensor, ...]
    global_slot_mappings: torch.Tensor
    hidden_restore_idx: torch.Tensor
    local_num_tokens_after_padding: int


class AscendPCPManager(PCPManager):
    """PCP manager that refreshes Ascend-only local-batch metadata."""

    @staticmethod
    def validate_config(
        vllm_config: VllmConfig,
        supports_mm_inputs: bool,
    ) -> None:
        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        if cudagraph_mode.has_full_cudagraphs():
            if cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
                raise NotImplementedError("MRV2 PCP supports FULL_DECODE_ONLY CUDA graphs only.")
            vllm_config = copy(vllm_config)
            vllm_config.compilation_config = copy(vllm_config.compilation_config)
            vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        speculative_config = vllm_config.speculative_config
        if (
            speculative_config is not None
            and vllm_config.parallel_config.prefill_context_parallel_size > 1
        ):
            if speculative_config.method not in ("mtp", "eagle3"):
                raise NotImplementedError(
                    "Ascend MRV2 PCP supports speculative decoding only with "
                    "MTP and Eagle3."
                )
            if not cudagraph_mode.has_full_cudagraphs():
                vllm_config = copy(vllm_config)
            # Upstream PCP still rejects every speculative method. Ascend
            # validates the supported methods above and owns their adaptation.
            vllm_config.speculative_config = None

        PCPManager.validate_config(vllm_config, supports_mm_inputs)

    def partition_batch(self, input_batch: AscendInputBatch) -> AscendInputBatch:
        """Partition the batch and update Ascend-specific local metadata."""
        global_batch = input_batch
        global_draft_counts = global_batch.num_draft_tokens_per_req
        if global_batch.num_draft_tokens > 0:
            if global_draft_counts is None:
                raise RuntimeError(
                    "PCP speculative decoding requires per-request draft token counts."
                )
            global_draft_counts = np.asarray(global_draft_counts, dtype=np.int32)
            if global_draft_counts.shape != (global_batch.num_reqs,):
                raise RuntimeError(
                    "PCP speculative draft counts must match the global request "
                    f"count: {global_draft_counts.shape} != "
                    f"({global_batch.num_reqs},)."
                )
            if np.any(global_draft_counts[global_batch.is_prefilling_np] != 0):
                raise RuntimeError(
                    "PCP speculative decoding does not support draft tokens on "
                    "prefill requests."
                )
            # Let upstream build the ordinary PCP rank-local layout. Draft
            # counts are restored below after its unsupported-method guard.
            input_batch = replace(
                global_batch,
                num_draft_tokens=0,
                num_draft_tokens_per_req=None,
            )

        local_batch = super().partition_batch(input_batch)
        assert isinstance(local_batch, AscendInputBatch)

        if global_draft_counts is not None:
            # Upstream rewrites the local decode tokens while constructing its
            # non-spec logits layout. Restore the already prepared K+1 target
            # inputs from the authoritative global batch.
            self._global_batch = global_batch
            assert self._padded_gather_idx is not None
            local_num_tokens_padded = local_batch.num_tokens_after_padding
            rank_token_start = self.pcp_rank * local_num_tokens_padded
            local_gather_idx = self._padded_gather_idx[
                rank_token_start : rank_token_start + local_num_tokens_padded
            ]
            torch.index_select(
                global_batch.input_ids,
                0,
                local_gather_idx,
                out=local_batch.input_ids[:local_num_tokens_padded],
            )
            draft_count_by_req = dict(
                zip(global_batch.req_ids, global_draft_counts, strict=True)
            )
            local_draft_counts = np.fromiter(
                (draft_count_by_req[req_id] for req_id in local_batch.req_ids),
                dtype=np.int32,
                count=local_batch.num_reqs,
            )
            local_batch = replace(
                local_batch,
                num_draft_tokens_per_req=local_draft_counts,
            )

        # PCP builds the local layout from actual tokens, but a FULL decode
        # graph replays a fixed padded layout on every rank.
        graph_num_tokens = global_batch.num_tokens_after_padding
        graph_num_reqs = global_batch.num_reqs_after_padding
        is_decode_only = not bool(global_batch.is_prefilling_np.any())
        if is_decode_only and graph_num_tokens > local_batch.num_tokens_after_padding:
            assert self._input_buffers is not None
            input_buffers = self._input_buffers
            actual_tokens = local_batch.num_tokens
            actual_reqs = local_batch.num_reqs
            if graph_num_tokens > input_buffers.max_num_tokens:
                raise RuntimeError(
                    "PCP graph token count exceeds the local input buffer: "
                    f"{graph_num_tokens} > {input_buffers.max_num_tokens}."
                )
            if graph_num_reqs > input_buffers.max_num_reqs:
                raise RuntimeError(
                    "PCP graph request count exceeds the local input buffer: "
                    f"{graph_num_reqs} > {input_buffers.max_num_reqs}."
                )
            input_buffers.input_ids[actual_tokens:graph_num_tokens].zero_()
            input_buffers.positions[actual_tokens:graph_num_tokens].zero_()
            input_buffers.is_padding[actual_tokens:graph_num_tokens].fill_(True)
            input_buffers.seq_lens[actual_reqs:graph_num_reqs].zero_()

            # Decode requests are replicated on every PCP rank, so the global
            # FULL-graph query layout is also the authoritative rank-local
            # layout, including any FIA dummy request.
            graph_query_start_loc_np = global_batch.query_start_loc_np[
                : graph_num_reqs + 1
            ]
            async_copy_to_gpu(
                graph_query_start_loc_np,
                out=input_buffers.query_start_loc[: graph_num_reqs + 1],
            )

            # Graph padding has no RankSegment, so _build_batch_layout does
            # not initialize the corresponding hidden restore indices.
            assert self._hidden_restore_idx is not None
            self._hidden_restore_idx[
                global_batch.num_tokens : graph_num_tokens
            ].zero_()
            seq_lens_cpu_upper_bound = torch.zeros(
                graph_num_reqs,
                dtype=local_batch.seq_lens_cpu_upper_bound.dtype,
            )
            seq_lens_cpu_upper_bound[:actual_reqs].copy_(local_batch.seq_lens_cpu_upper_bound[:actual_reqs])
            local_batch = replace(  # type: ignore[call-arg]
                local_batch,
                num_reqs_after_padding=graph_num_reqs,
                num_tokens_after_padding=graph_num_tokens,
                query_start_loc=input_buffers.query_start_loc[: graph_num_reqs + 1],
                query_start_loc_np=graph_query_start_loc_np,
                seq_lens=input_buffers.seq_lens[:graph_num_reqs],
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                input_ids=input_buffers.input_ids[:graph_num_tokens],
                positions=input_buffers.positions[:graph_num_tokens],
                is_padding=input_buffers.is_padding[:graph_num_tokens],
            )

        local_seq_lens_np = (
            local_batch.num_computed_tokens_np
            + local_batch.num_scheduled_tokens
        )
        if local_batch.num_reqs_after_padding > local_batch.num_reqs:
            padded_seq_lens_np = np.zeros(
                local_batch.num_reqs_after_padding,
                dtype=local_seq_lens_np.dtype,
            )
            padded_seq_lens_np[: local_batch.num_reqs] = local_seq_lens_np
            local_seq_lens_np = padded_seq_lens_np

        local_batch.seq_lens_np = local_seq_lens_np
        return local_batch

    def restore_hidden_state_buffer(self, hidden_states: torch.Tensor) -> None:
        """Restore a model-owned rank-local buffer to the global PCP layout."""
        assert self._padded_gather_idx is not None
        local_num_tokens_padded = self._padded_gather_idx.shape[0] // self.pcp_world_size
        restored_hidden_states = self.restore_hidden_states(hidden_states[:local_num_tokens_padded])
        hidden_states[: restored_hidden_states.shape[0]].copy_(restored_hidden_states)

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Use capture-safe dummy metadata without changing runtime prep."""
        if not input_batch.is_dummy:
            return super().prepare_attn(input_batch)

        assert self._local_block_tables is not None
        num_reqs = input_batch.num_reqs_after_padding
        for block_table in self._local_block_tables:
            block_table[:num_reqs].zero_()
        return (
            tuple(block_table[:num_reqs] for block_table in self._local_block_tables),
            self.get_dummy_slot_mappings(input_batch.num_tokens_after_padding),
        )

    def prepare_slot_mappings(self) -> torch.Tensor:
        """Pad PCP slot mappings to the fixed FULL-decode graph layout."""
        slot_mappings = super().prepare_slot_mappings()
        assert self._global_batch is not None
        graph_num_tokens = self._global_batch.num_tokens_after_padding
        is_decode_only = not bool(self._global_batch.is_prefilling_np.any())
        if not is_decode_only or graph_num_tokens <= self._global_batch.num_tokens:
            return slot_mappings

        assert self._gathered_kv_slot_mappings is not None
        graph_num_slots = graph_num_tokens * self.pcp_world_size
        self._gathered_kv_slot_mappings[:, slot_mappings.shape[1] : graph_num_slots].fill_(-1)
        return self._gathered_kv_slot_mappings[:, :graph_num_slots]

    def build_attention_context(
        self,
        input_batch: AscendInputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
    ) -> AscendPCPAttentionContext:
        """Build the PCP context consumed by attention metadata builders."""
        if input_batch.is_dummy:
            local_num_tokens_after_padding = input_batch.num_tokens
            restore_start = self.pcp_rank * local_num_tokens_after_padding
            return AscendPCPAttentionContext(
                global_batch=input_batch,
                global_block_tables=block_tables,
                global_slot_mappings=slot_mappings.view(
                    slot_mappings.shape[0],
                    self.pcp_world_size,
                    local_num_tokens_after_padding,
                )[:, self.pcp_rank],
                hidden_restore_idx=torch.arange(
                    restore_start,
                    restore_start + local_num_tokens_after_padding,
                    device=self.device,
                ),
                local_num_tokens_after_padding=local_num_tokens_after_padding,
            )

        global_batch = self._global_batch
        return AscendPCPAttentionContext(
            global_batch=global_batch,
            global_block_tables=self._block_tables.gather_block_tables(
                global_batch.idx_mapping,
                global_batch.num_reqs_after_padding,
            ),
            global_slot_mappings=self._global_batch_slot_mappings[:, : global_batch.num_tokens],
            hidden_restore_idx=self._hidden_restore_idx,
            local_num_tokens_after_padding=input_batch.num_tokens_after_padding,
        )
