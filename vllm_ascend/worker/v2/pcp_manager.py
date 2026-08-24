# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/pcp_manager.py
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

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import prepare_pos_seq_lens
from vllm.v1.worker.gpu.pcp_manager import PCPManager, RankSegment
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

SUPPORTED_PCP_SPECULATIVE_METHODS = frozenset({"mtp", "eagle3"})

logger = init_logger(__name__)


class AscendPCPManager(PCPManager):
    """PCP manager that refreshes Ascend-only local-batch metadata."""

    @staticmethod
    def validate_config(
        vllm_config: VllmConfig,
        supports_mm_inputs: bool,
    ) -> None:
        """Validate the Ascend MRV2 MLA and GQA PCP implementations."""
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config
        if parallel_config.prefill_context_parallel_size <= 1:
            return

        speculative_config = vllm_config.speculative_config
        if speculative_config is not None and speculative_config.method not in SUPPORTED_PCP_SPECULATIVE_METHODS:
            raise NotImplementedError("Ascend MRV2 PCP supports speculative decoding only with MTP and Eagle3.")
        if parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError("Ascend MRV2 does not support PCP and DCP simultaneously yet.")
        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError("Ascend MRV2 PCP does not support PP yet.")
        if model_config.is_encoder_decoder:
            raise NotImplementedError("Ascend MRV2 PCP does not support encoder-decoder models yet.")
        if supports_mm_inputs:
            raise NotImplementedError("Ascend MRV2 PCP does not support MM inputs yet.")
        if vllm_config.lora_config is not None:
            raise NotImplementedError("Ascend MRV2 PCP does not support LoRA yet.")

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        device: torch.device,
        vllm_config: VllmConfig | None = None,
        req_states: RequestState | None = None,
        max_num_reqs: int | None = None,
        max_num_tokens: int | None = None,
        block_tables: BlockTables | None = None,
        dcp_world_size: int = 1,
        dcp_rank: int = 0,
        cp_interleave: int = 1,
    ) -> None:
        super().__init__(
            pcp_world_size,
            pcp_rank,
            device,
            req_states=req_states,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            block_tables=block_tables,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            cp_interleave=cp_interleave,
        )
        self.vllm_config = vllm_config

    def partition_batch(
        self,
        input_batch: AscendInputBatch,
    ) -> AscendInputBatch:
        """Partition PCP batches with Ascend speculative-decoding support.

        Adapted from vLLM PCPManager.partition_batch at
        82db054a5c6e7bb19ffe38133dbb127455bb7e81.
        """
        assert self._req_states is not None
        assert self._input_buffers is not None
        assert self.vllm_config is not None
        input_buffers = self._input_buffers

        graph_num_reqs = input_batch.num_reqs_after_padding
        graph_num_tokens = input_batch.num_tokens_after_padding
        is_decode_only = not np.any(input_batch.is_prefilling_np)

        global_batch = input_batch
        self._global_batch = global_batch

        num_scheduled_tokens = global_batch.num_scheduled_tokens
        num_computed_tokens = global_batch.num_computed_tokens_np
        is_prefilling = global_batch.is_prefilling_np

        global_draft_counts = global_batch.num_draft_tokens_per_req
        if global_batch.num_draft_tokens > 0 and global_draft_counts is None:
            raise RuntimeError("PCP speculative decoding requires per-request draft token counts.")
        if global_draft_counts is not None:
            global_draft_counts = np.asarray(global_draft_counts, dtype=np.int32)
            if global_draft_counts.shape != (global_batch.num_reqs,):
                raise RuntimeError(
                    "PCP speculative draft counts must match the global request "
                    f"count: {global_draft_counts.shape} != "
                    f"({global_batch.num_reqs},)."
                )
            if np.any(global_draft_counts[is_prefilling] != 0):
                raise RuntimeError("PCP speculative decoding does not support draft tokens on prefill requests.")

        segments_by_rank, per_rank_num_tokens = self._build_batch_layout(
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
            global_batch.query_start_loc_np,
        )
        if graph_num_tokens > global_batch.num_tokens:
            # Graph padding has no RankSegment, so its restore indices are
            # otherwise left uninitialized by the upstream layout builder.
            assert self._hidden_restore_idx is not None
            assert self._hidden_restore_idx.shape[0] == graph_num_tokens
            self._hidden_restore_idx[global_batch.num_tokens : graph_num_tokens].zero_()

        local_segments = segments_by_rank[self.pcp_rank]
        if not local_segments:
            local_segments = [
                RankSegment(
                    global_batch_req_idx=0,
                    global_batch_slice=slice(0, 0),
                    rank_local_batch_slice=slice(0, 0),
                )
            ]

        num_local_reqs = len(local_segments)
        if num_local_reqs > input_buffers.max_num_reqs:
            raise RuntimeError(
                "PCP local request count exceeds the MRV2 input buffer size: "
                f"{num_local_reqs} > {input_buffers.max_num_reqs}."
            )

        local_to_global_batch_req_idx_np = np.fromiter(
            (segment.global_batch_req_idx for segment in local_segments),
            dtype=np.int32,
            count=num_local_reqs,
        )
        local_start_pos_np = np.fromiter(
            (
                num_computed_tokens[segment.global_batch_req_idx]
                + segment.global_batch_slice.start
                - global_batch.query_start_loc_np[segment.global_batch_req_idx]
                for segment in local_segments
            ),
            dtype=np.int32,
            count=num_local_reqs,
        )
        local_num_scheduled_tokens = np.fromiter(
            (segment.num_tokens for segment in local_segments),
            dtype=np.int32,
            count=num_local_reqs,
        )
        local_to_global_req_idx_np = global_batch.idx_mapping_np[local_to_global_batch_req_idx_np]
        local_req_ids = [
            global_batch.req_ids[global_batch_req_idx] for global_batch_req_idx in local_to_global_batch_req_idx_np
        ]

        num_local_tokens = int(local_num_scheduled_tokens.sum())
        num_local_tokens_padded = max(per_rank_num_tokens)
        fresh_prefills = int(np.count_nonzero(is_prefilling & (num_computed_tokens == 0)))
        continued_prefills = int(np.count_nonzero(is_prefilling & (num_computed_tokens > 0)))
        logger.debug(
            "PCP batch: rank=%d global_batch_reqs=%d fresh_prefills=%d "
            "continued_prefills=%d decodes=%d local_reqs=%d "
            "local_tokens=%d per_rank_tokens=%s",
            self.pcp_rank,
            global_batch.num_reqs,
            fresh_prefills,
            continued_prefills,
            global_batch.num_reqs - fresh_prefills - continued_prefills,
            num_local_reqs,
            num_local_tokens,
            per_rank_num_tokens,
        )
        if num_local_tokens_padded > input_buffers.max_num_tokens:
            raise RuntimeError(
                "PCP local token count exceeds the MRV2 input buffer size: "
                f"{num_local_tokens_padded} > {input_buffers.max_num_tokens}."
            )
        rank_token_start = self.pcp_rank * num_local_tokens_padded
        assert self._padded_gather_idx is not None
        local_gather_idx = self._padded_gather_idx[rank_token_start : rank_token_start + num_local_tokens_padded]
        torch.index_select(
            global_batch.input_ids,
            0,
            local_gather_idx,
            out=input_buffers.input_ids[:num_local_tokens_padded],
        )

        local_query_start_loc_np = np.empty(input_buffers.max_num_reqs + 1, dtype=np.int32)
        local_query_start_loc_np[0] = 0
        local_query_start_loc_out = local_query_start_loc_np[1 : num_local_reqs + 1]
        np.cumsum(local_num_scheduled_tokens, out=local_query_start_loc_out)
        local_query_start_loc_np[num_local_reqs + 1 :] = num_local_tokens
        async_copy_to_gpu(
            local_query_start_loc_np,
            out=input_buffers.query_start_loc,
        )
        local_query_start_loc = input_buffers.query_start_loc[: num_local_reqs + 1]

        local_to_global_req_idx = async_copy_to_gpu(local_to_global_req_idx_np, device=self.device)
        local_start_pos = async_copy_to_gpu(
            local_start_pos_np,
            device=self.device,
        )

        assert self._local_req_idx is not None
        prepare_pos_seq_lens(
            self._local_req_idx[:num_local_reqs],
            local_query_start_loc,
            local_start_pos,
            input_buffers.positions,
            input_buffers.seq_lens[:num_local_reqs],
        )
        seq_lens = input_buffers.seq_lens[:num_local_reqs]
        is_padding = input_buffers.is_padding[:num_local_tokens_padded]
        is_padding[:num_local_tokens].fill_(False)
        is_padding[num_local_tokens:].fill_(True)
        if num_local_tokens_padded > num_local_tokens:
            input_buffers.input_ids[:num_local_tokens_padded].masked_fill_(is_padding, 0)
            input_buffers.positions[:num_local_tokens_padded].masked_fill_(is_padding, 0)

        local_draft_counts = None
        if global_draft_counts is not None:
            local_draft_counts = global_draft_counts[local_to_global_batch_req_idx_np].copy()
            if num_local_tokens == 0:
                local_draft_counts.fill(0)

        # The global batch already contains sampled and draft token IDs. Decode
        # rows are gathered whole, so do not rewrite the local token buffer.
        # Sampling later switches back to the saved global batch; these local
        # logits fields are ordinary PCP placeholders only.
        total_num_logits = num_local_reqs if num_local_tokens > 0 else 0
        if total_num_logits > 0:
            cu_num_logits_np = np.arange(num_local_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(
                num_local_reqs + 1,
                device=self.device,
                dtype=torch.int32,
            )
            logits_indices = (local_query_start_loc[1:] - 1).to(torch.int64)
        else:
            cu_num_logits_np = np.zeros(num_local_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.zeros(
                num_local_reqs + 1,
                device=self.device,
                dtype=torch.int32,
            )
            logits_indices = torch.empty(0, device=self.device, dtype=torch.int64)

        local_prefill_len_np = global_batch.prefill_len_np[local_to_global_batch_req_idx_np]
        local_num_computed_prefill_tokens_np = np.minimum(
            local_start_pos_np,
            local_prefill_len_np,
        )
        local_is_prefilling_np = local_num_computed_prefill_tokens_np < local_prefill_len_np
        seq_lens_cpu_upper_bound_np = np.zeros(
            num_local_reqs,
            dtype=np.int32,
        )
        seq_lens_cpu_upper_bound_np[:] = local_start_pos_np + local_num_scheduled_tokens

        dcp_local_seq_lens = None
        if self.dcp_world_size > 1:
            prepare_dcp_local_seq_lens(
                input_buffers.dcp_local_seq_lens,
                seq_lens,
                num_local_reqs,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_interleave,
            )
            dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[:num_local_reqs]

        local_batch = replace(
            global_batch,
            req_ids=local_req_ids,
            num_reqs=num_local_reqs,
            num_reqs_after_padding=num_local_reqs,
            idx_mapping=local_to_global_req_idx,
            idx_mapping_np=local_to_global_req_idx_np,
            expanded_idx_mapping=local_to_global_req_idx,
            expanded_local_pos=torch.zeros(
                num_local_reqs,
                dtype=torch.int32,
                device=self.device,
            ),
            num_scheduled_tokens=local_num_scheduled_tokens,
            num_tokens=num_local_tokens,
            num_tokens_after_padding=num_local_tokens_padded,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=local_query_start_loc,
            query_start_loc_np=local_query_start_loc_np[: num_local_reqs + 1],
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_cpu_upper_bound_np),
            dcp_local_seq_lens=dcp_local_seq_lens,
            num_computed_tokens_np=local_start_pos_np,
            prefill_len_np=local_prefill_len_np,
            num_computed_prefill_tokens_np=(local_num_computed_prefill_tokens_np),
            is_prefilling_np=local_is_prefilling_np,
            max_seq_len_np=global_batch.max_seq_len_np[local_to_global_batch_req_idx_np]
            if global_batch.max_seq_len_np is not None
            else None,
            input_ids=input_buffers.input_ids[:num_local_tokens_padded],
            positions=input_buffers.positions[:num_local_tokens_padded],
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            prompt_lens=None,
        )

        local_seq_lens_np = local_batch.num_computed_tokens_np + local_batch.num_scheduled_tokens
        if is_decode_only:
            local_batch.attn_state = input_batch.attn_state
            if local_batch.attn_state is None:
                local_batch.attn_state = AscendAttentionState.DecodeOnly
            return self._pad_decode_batch_for_full_graph(
                local_batch,
                graph_num_reqs,
                graph_num_tokens,
                local_seq_lens_np,
            )

        local_num_valid_tokens = local_batch.num_scheduled_tokens
        if local_draft_counts is not None:
            local_num_valid_tokens = local_batch.num_scheduled_tokens - local_draft_counts
        local_batch.attn_state = build_attn_state(
            self.vllm_config,
            local_seq_lens_np,
            local_batch.num_reqs,
            local_batch.num_scheduled_tokens,
            local_num_valid_tokens,
        )
        local_batch.seq_lens_np = local_seq_lens_np
        return local_batch

    def restore_hidden_state_buffer(
        self,
        hidden_states: torch.Tensor,
    ) -> None:
        """Restore a model-owned rank-local buffer to the global PCP layout."""
        assert self._padded_gather_idx is not None
        local_num_tokens_padded = self._padded_gather_idx.shape[0] // self.pcp_world_size
        restored_hidden_states = self.restore_hidden_states(hidden_states[:local_num_tokens_padded])
        hidden_states[: restored_hidden_states.shape[0]].copy_(restored_hidden_states)

    def _pad_decode_batch_for_full_graph(
        self,
        local_batch: AscendInputBatch,
        graph_num_reqs: int,
        graph_num_tokens: int,
        local_seq_lens_np: np.ndarray,
    ) -> AscendInputBatch:
        """Pad a rank-local decode batch to the selected full-graph shape."""
        num_reqs = local_batch.num_reqs
        num_tokens = local_batch.num_tokens
        if graph_num_reqs < num_reqs or graph_num_tokens < num_tokens:
            raise RuntimeError(
                "PCP graph shape is smaller than the rank-local decode batch: "
                f"requests {graph_num_reqs} < {num_reqs} or "
                f"tokens {graph_num_tokens} < {num_tokens}."
            )

        num_padding_reqs = graph_num_reqs - num_reqs
        num_padding_tokens = graph_num_tokens - num_tokens
        decode_query_len = getattr(self.vllm_config, "num_speculative_tokens", 0) + 1
        uses_per_request_padding = num_padding_tokens == num_padding_reqs * decode_query_len
        uses_single_fia_dummy = num_padding_reqs == 1 and num_padding_tokens > 0
        if not (uses_per_request_padding or uses_single_fia_dummy):
            raise RuntimeError(
                "PCP FULL_DECODE_ONLY requires either a uniform decode query "
                "per padded request or one FIA dummy request for all padding "
                "tokens: "
                f"{num_padding_tokens} tokens for {num_padding_reqs} requests."
            )

        # Full-graph model inputs keep using the model runner's original
        # buffers, while PCP attention metadata uses the rank-local buffers.
        # Clear both views so a smaller replay cannot observe stale padding
        # left by a previously larger decode batch.
        # Clear the full image buffer to prevent dirty data from polluting the next image.
        global_batch = self._global_batch
        assert global_batch is not None
        global_batch.input_ids[num_tokens:graph_num_tokens].zero_()
        global_batch.positions[num_tokens:graph_num_tokens].zero_()
        global_batch.is_padding[:num_tokens].fill_(False)
        global_batch.is_padding[num_tokens:graph_num_tokens].fill_(True)

        if num_padding_reqs == 0:
            local_batch.seq_lens_np = local_seq_lens_np
            return local_batch

        input_buffers = self._input_buffers
        assert input_buffers is not None
        if graph_num_reqs > input_buffers.max_num_reqs:
            raise RuntimeError(
                "PCP graph request count exceeds the local input buffer capacity: "
                f"{graph_num_reqs} > {input_buffers.max_num_reqs}."
            )
        if graph_num_tokens > input_buffers.max_num_tokens:
            raise RuntimeError(
                "PCP graph token count exceeds the local input buffer capacity: "
                f"{graph_num_tokens} > {input_buffers.max_num_tokens}."
            )

        input_buffers.input_ids[num_tokens:graph_num_tokens].zero_()
        input_buffers.positions[num_tokens:graph_num_tokens].zero_()
        input_buffers.seq_lens[num_reqs:graph_num_reqs].zero_()
        input_buffers.is_padding[:num_tokens].fill_(False)
        input_buffers.is_padding[num_tokens:graph_num_tokens].fill_(True)

        query_start_loc_buffer_np = np.full(
            input_buffers.max_num_reqs + 1,
            graph_num_tokens,
            dtype=np.int32,
        )
        query_start_loc_buffer_np[: num_reqs + 1] = local_batch.query_start_loc_np
        if uses_per_request_padding:
            query_start_loc_buffer_np[num_reqs + 1 : graph_num_reqs + 1] = num_tokens + decode_query_len * np.arange(
                1,
                num_padding_reqs + 1,
                dtype=np.int32,
            )
        else:
            query_start_loc_buffer_np[num_reqs + 1 : graph_num_reqs + 1] = graph_num_tokens
        async_copy_to_gpu(
            query_start_loc_buffer_np,
            out=input_buffers.query_start_loc,
        )
        query_start_loc_np = query_start_loc_buffer_np[: graph_num_reqs + 1]

        padded_seq_lens_np = np.zeros(graph_num_reqs, dtype=np.int32)
        padded_seq_lens_np[:num_reqs] = local_seq_lens_np
        seq_lens_cpu_upper_bound = torch.zeros(graph_num_reqs, dtype=torch.int32)
        seq_lens_cpu_upper_bound[:num_reqs].copy_(local_batch.seq_lens_cpu_upper_bound)

        return replace(
            local_batch,
            num_reqs_after_padding=graph_num_reqs,
            num_tokens_after_padding=graph_num_tokens,
            query_start_loc=input_buffers.query_start_loc[: graph_num_reqs + 1],
            query_start_loc_np=query_start_loc_np,
            seq_lens=input_buffers.seq_lens[:graph_num_reqs],
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            input_ids=input_buffers.input_ids[:graph_num_tokens],
            positions=input_buffers.positions[:graph_num_tokens],
            is_padding=input_buffers.is_padding[:graph_num_tokens],
            seq_lens_np=padded_seq_lens_np,
        )

    def prepare_dummy_attn(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Prepare capture inputs using the same buffers as graph replay."""
        assert self._local_block_tables is not None
        for block_table in self._local_block_tables:
            block_table[:num_reqs].zero_()
        return (
            tuple(block_table[:num_reqs] for block_table in self._local_block_tables),
            self.get_dummy_slot_mappings(num_tokens),
        )

    def prepare_slot_mappings(self) -> torch.Tensor:
        slot_mappings = super().prepare_slot_mappings()
        assert self._global_batch is not None
        assert self._gathered_kv_slot_mappings is not None

        global_batch = self._global_batch
        if np.any(global_batch.is_prefilling_np):
            return slot_mappings

        graph_num_tokens = global_batch.num_tokens_after_padding
        if graph_num_tokens <= global_batch.num_tokens:
            return slot_mappings

        num_expanded_tokens = slot_mappings.shape[1]
        graph_num_expanded_tokens = graph_num_tokens * self.pcp_world_size
        if graph_num_expanded_tokens > self._gathered_kv_slot_mappings.shape[1]:
            raise RuntimeError(
                "PCP graph slot mapping exceeds the persistent buffer capacity: "
                f"{graph_num_expanded_tokens} > "
                f"{self._gathered_kv_slot_mappings.shape[1]}."
            )
        self._gathered_kv_slot_mappings[:, num_expanded_tokens:graph_num_expanded_tokens].fill_(PAD_SLOT_ID)
        return self._gathered_kv_slot_mappings[:, :graph_num_expanded_tokens]
