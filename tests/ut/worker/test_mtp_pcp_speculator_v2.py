# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionPCPMetadata,
    AscendMetadata,
)
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.spec_decode.autoregressive import (
    speculator as speculator_module,
)
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import (
    AscendEagleSpeculator,
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
    input_batch.query_start_loc = torch.arange(5, dtype=torch.int32)
    input_batch.query_start_loc_np = np.arange(5, dtype=np.int32)
    input_batch.seq_lens = torch.arange(4, dtype=torch.int32)
    input_batch.seq_lens_cpu_upper_bound = torch.arange(4, dtype=torch.int32)
    input_batch.input_ids = torch.arange(8, dtype=torch.int32)
    input_batch.positions = torch.arange(8, dtype=torch.int64)
    input_batch.is_padding = torch.zeros(8, dtype=torch.bool)
    input_batch.seq_lens_np = np.arange(4, dtype=np.int32)
    return input_batch


def test_mtp_uses_draft_groups_only_when_target_pcp_is_active() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.attn_groups = MagicMock()
    speculator.target_attn_groups = MagicMock()
    speculator.pcp_manager = None
    assert speculator.draft_prefill_attn_groups is speculator.target_attn_groups

    speculator.pcp_manager = MagicMock()
    assert speculator.draft_prefill_attn_groups is speculator.attn_groups


@pytest.mark.parametrize("pcp_size", [1, 2])
def test_autoregressive_speculator_reuses_target_config(pcp_size: int) -> None:
    vllm_config = MagicMock()
    vllm_config.parallel_config.prefill_context_parallel_size = pcp_size
    device = MagicMock()

    with (
        patch.object(
            speculator_module.AutoRegressiveSpeculator,
            "__init__",
            side_effect=RuntimeError("stop after config"),
        ) as base_init,
        pytest.raises(RuntimeError, match="stop after config"),
    ):
        speculator = object.__new__(AscendMTPSpeculator)
        speculator_module.AscendAutoRegressiveSpeculator.__init__(
            speculator,
            vllm_config,
            device,
        )

    base_init.assert_called_once_with(vllm_config, device)


@pytest.mark.parametrize(
    ("draft_mode", "enforce_eager", "expected_mode"),
    [
        (CUDAGraphMode.NONE, False, CUDAGraphMode.NONE),
        (
            CUDAGraphMode.FULL_DECODE_ONLY,
            False,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ),
        (CUDAGraphMode.FULL_DECODE_ONLY, True, CUDAGraphMode.NONE),
    ],
)
def test_autoregressive_graph_manager_respects_draft_config(
    draft_mode: CUDAGraphMode,
    enforce_eager: bool,
    expected_mode: CUDAGraphMode,
) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.vllm_config = MagicMock()
    speculator.vllm_config.compilation_config.cudagraph_mode = draft_mode
    speculator.speculative_config = MagicMock(enforce_eager=enforce_eager)
    speculator.prefill_cudagraph_manager = MagicMock()
    speculator.decode_cudagraph_manager = MagicMock()
    speculator.update_stream = MagicMock()

    with patch.object(
        speculator_module.AutoRegressiveSpeculator,
        "init_cudagraph_manager",
    ) as parent_init:
        speculator.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)

    parent_init.assert_called_once_with(expected_mode)


def test_set_attn_disables_pcp_only_for_draft_attention() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    draft_vllm_config = MagicMock()
    speculator.vllm_config = draft_vllm_config
    draft_layer_name = "draft.mla"
    speculator.draft_attn_layer_names = {draft_layer_name}

    gqa_builder = speculator_module.AscendAttentionPCPMetadataBuilder.__new__(
        speculator_module.AscendAttentionPCPMetadataBuilder
    )
    gqa_builder.set_pcp_enabled(True)
    mla_builder = speculator_module.AscendMLAPCPMetadataBuilder.__new__(speculator_module.AscendMLAPCPMetadataBuilder)
    mla_builder.set_pcp_enabled(True)
    draft_group = SimpleNamespace(metadata_builders=[gqa_builder, mla_builder])

    target_builder = speculator_module.AscendAttentionPCPMetadataBuilder.__new__(
        speculator_module.AscendAttentionPCPMetadataBuilder
    )
    target_builder.set_pcp_enabled(True)
    target_attn_groups = [[SimpleNamespace(metadata_builders=[target_builder])]]

    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [SimpleNamespace(layer_names=[draft_layer_name])]
    draft_layer = MagicMock(spec=speculator_module.MLAAttention)
    draft_layer.use_pcp = True
    draft_layer.get_attn_backend.return_value = speculator_module.AscendMLABackend
    config_context = MagicMock()
    model_state = MagicMock()
    block_tables = MagicMock()
    target_input_buffers = MagicMock()

    def set_draft_groups(*_args, **_kwargs) -> None:
        speculator.attn_groups = [[draft_group]]

    with (
        patch.object(
            speculator_module,
            "set_current_vllm_config",
            return_value=config_context,
        ) as set_current_config,
        patch.object(
            speculator_module.AutoRegressiveSpeculator,
            "set_attn",
            side_effect=set_draft_groups,
        ) as parent_set_attn,
        patch.object(
            speculator_module,
            "get_layers_from_vllm_config",
            return_value={draft_layer_name: draft_layer},
        ),
    ):
        speculator.set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )

    set_current_config.assert_called_once_with(draft_vllm_config)
    config_context.__enter__.assert_called_once_with()
    config_context.__exit__.assert_called_once()
    parent_set_attn.assert_called_once_with(
        model_state,
        kv_cache_config,
        block_tables,
        target_input_buffers,
        target_attn_groups,
    )
    assert not gqa_builder.use_pcp
    assert gqa_builder.metadata_cls is AscendMetadata
    assert not mla_builder.use_pcp
    assert mla_builder.metadata_cls is AscendMLAMetadata
    assert target_builder.use_pcp
    assert target_builder.metadata_cls is AscendAttentionPCPMetadata
    assert not draft_layer.use_pcp
    assert speculator.attn_backends == {draft_layer_name: speculator_module.AscendMLABackend}
    assert speculator.attn_architecture == "MLA"


@pytest.mark.parametrize(
    ("num_reqs_after_padding", "num_tokens_after_padding", "cudagraph_mode"),
    [
        (2, 6, CUDAGraphMode.NONE),
        (4, 8, CUDAGraphMode.FULL),
    ],
)
def test_prepare_replicated_pcp_attn_uses_speculator_owned_global_inputs(
    num_reqs_after_padding: int,
    num_tokens_after_padding: int,
    cudagraph_mode: CUDAGraphMode,
) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.pcp_manager = MagicMock()
    speculator.block_tables = MagicMock()
    speculator.model_state = MagicMock()
    speculator.attn_groups = MagicMock()
    speculator.target_attn_groups = MagicMock()
    speculator.kv_cache_config = MagicMock()

    input_batch = MagicMock(spec=AscendInputBatch)
    input_batch.num_reqs = 2
    input_batch.num_reqs_after_padding = num_reqs_after_padding
    input_batch.num_tokens = 6
    input_batch.num_tokens_after_padding = num_tokens_after_padding
    input_batch.idx_mapping = torch.tensor([3, 7], dtype=torch.int32)
    input_batch.query_start_loc = torch.tensor([0, 3, 6, 8, 8][: num_reqs_after_padding + 1], dtype=torch.int32)
    input_batch.positions = torch.arange(num_tokens_after_padding, dtype=torch.int64)

    block_tables = (MagicMock(),)
    global_slot_mapping = torch.arange(num_tokens_after_padding).unsqueeze(0)
    attn_metadata = MagicMock()
    slot_mappings = MagicMock()
    speculator.block_tables.gather_block_tables.return_value = block_tables
    speculator.block_tables.compute_slot_mappings.return_value = global_slot_mapping
    speculator.model_state.prepare_attn.return_value = attn_metadata

    with patch.object(
        speculator_module,
        "build_slot_mappings_by_layer",
        return_value=slot_mappings,
    ) as build_slot_mappings:
        result = speculator._prepare_replicated_pcp_attn(input_batch)

    assert result[0] is attn_metadata
    assert result[1] is slot_mappings
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
    speculator.model_state.prepare_attn.assert_called_once_with(
        input_batch,
        cudagraph_mode,
        block_tables,
        global_slot_mapping,
        speculator.attn_groups,
        speculator.kv_cache_config,
    )
    build_slot_mappings.assert_called_once_with(
        global_slot_mapping,
        speculator.kv_cache_config,
    )


@pytest.mark.parametrize(
    ("method", "speculator_cls", "parent_cls"),
    [
        ("mtp", AscendMTPSpeculator, MTPSpeculator),
        ("eagle3", AscendEagleSpeculator, EagleSpeculator),
    ],
)
def test_speculator_uses_replicated_pcp_attention(
    method: str,
    speculator_cls,
    parent_cls,
) -> None:
    """Verify padded global proposal inputs for both MTP and Eagle3."""
    speculator = object.__new__(speculator_cls)
    speculator.method = method
    speculator.input_batch = None
    speculator.pcp_manager = MagicMock()

    input_batch = _make_padded_input_batch()
    last_hidden_states = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    local_attn_metadata = MagicMock()
    local_slot_mappings = MagicMock()
    global_attn_metadata = MagicMock()
    global_slot_mappings = MagicMock()
    local_aux_hidden_states = [
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
    ]
    global_aux_hidden_states = torch.arange(56, dtype=torch.float32).reshape(8, 7)
    aux_hidden_states = local_aux_hidden_states if method == "eagle3" else None
    speculator.pcp_manager.restore_hidden_states.return_value = global_aux_hidden_states

    with (
        patch.object(parent_cls, "propose", return_value=MagicMock()) as propose,
        patch.object(
            speculator_cls,
            "_prepare_replicated_pcp_attn",
            return_value=(global_attn_metadata, global_slot_mappings),
        ) as prepare_replicated_pcp_attn,
    ):
        speculator_cls.propose(
            speculator,
            input_batch,
            local_attn_metadata,
            local_slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

    prepare_replicated_pcp_attn.assert_called_once_with(input_batch)

    propose_args = propose.call_args.args
    assert propose_args[0] is input_batch
    assert propose_args[1] is global_attn_metadata
    assert propose_args[2] is global_slot_mappings

    torch.testing.assert_close(propose_args[3], last_hidden_states)

    if method == "eagle3":
        restore_args = speculator.pcp_manager.restore_hidden_states.call_args.args
        torch.testing.assert_close(
            restore_args[0],
            torch.cat(local_aux_hidden_states, dim=-1),
        )
        assert len(propose_args[4]) == 1
        torch.testing.assert_close(propose_args[4][0], global_aux_hidden_states)
    else:
        speculator.pcp_manager.restore_hidden_states.assert_not_called()
        assert propose_args[4] is None


def test_eagle3_pcp_dummy_run_keeps_local_inputs() -> None:
    speculator = object.__new__(AscendEagleSpeculator)
    speculator.method = "eagle3"
    speculator.input_batch = None
    speculator.pcp_manager = MagicMock()

    input_batch = MagicMock()
    local_attn_metadata = MagicMock()
    local_slot_mappings = MagicMock()
    local_aux_hidden_states = [MagicMock(), MagicMock()]

    with (
        patch.object(
            EagleSpeculator,
            "propose",
            return_value=MagicMock(),
        ) as propose,
        patch.object(
            AscendEagleSpeculator,
            "_prepare_replicated_pcp_attn",
        ) as prepare_replicated_pcp_attn,
    ):
        AscendEagleSpeculator.propose(
            speculator,
            input_batch,
            local_attn_metadata,
            local_slot_mappings,
            MagicMock(),
            local_aux_hidden_states,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            dummy_run=True,
        )

    prepare_replicated_pcp_attn.assert_not_called()
    speculator.pcp_manager.restore_hidden_states.assert_not_called()

    propose_args = propose.call_args.args
    assert propose_args[1] is local_attn_metadata
    assert propose_args[2] is local_slot_mappings
    assert propose_args[4] is local_aux_hidden_states


def test_eagle3_pcp_requires_aux_hidden_states() -> None:
    speculator = object.__new__(AscendEagleSpeculator)
    speculator.method = "eagle3"
    speculator.input_batch = None
    speculator.pcp_manager = MagicMock()

    input_batch = _make_padded_input_batch()

    with (
        patch.object(EagleSpeculator, "propose") as propose,
        patch.object(
            AscendEagleSpeculator,
            "_prepare_replicated_pcp_attn",
            return_value=(MagicMock(), MagicMock()),
        ),
        pytest.raises(
            RuntimeError,
            match="requires auxiliary target hidden states",
        ),
    ):
        AscendEagleSpeculator.propose(
            speculator,
            input_batch,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            None,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

    propose.assert_not_called()


@pytest.mark.parametrize(
    ("method", "speculator_cls", "parent_cls"),
    [
        ("mtp", AscendMTPSpeculator, MTPSpeculator),
        ("eagle3", AscendEagleSpeculator, EagleSpeculator),
    ],
)
def test_speculator_without_pcp_keeps_existing_proposal_inputs(
    method: str,
    speculator_cls,
    parent_cls,
) -> None:
    speculator = object.__new__(speculator_cls)
    speculator.method = method
    speculator.input_batch = None
    speculator.pcp_manager = None

    input_batch = MagicMock()
    local_attn_metadata = MagicMock()
    local_slot_mappings = MagicMock()
    local_aux_hidden_states = [MagicMock()] if method == "eagle3" else None

    with (
        patch.object(
            parent_cls,
            "propose",
            return_value=MagicMock(),
        ) as propose,
        patch.object(
            speculator_module,
            "build_slot_mappings_by_layer",
        ) as build_slot_mappings,
    ):
        speculator_cls.propose(
            speculator,
            input_batch,
            local_attn_metadata,
            local_slot_mappings,
            MagicMock(),
            local_aux_hidden_states,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

    build_slot_mappings.assert_not_called()
    propose_args = propose.call_args.args
    assert propose_args[1] is local_attn_metadata
    assert propose_args[2] is local_slot_mappings
    assert propose_args[4] is local_aux_hidden_states
