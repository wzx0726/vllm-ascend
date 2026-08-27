# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionPCPMetadata,
    AscendMetadata,
)
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import (
    AscendEagleSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
    AscendMTPSpeculator,
)

from vllm_ascend.worker.v2.spec_decode.autoregressive import (
    speculator as speculator_module,
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


def test_draft_runtime_config_uses_draft_parallel_topology() -> None:
    draft_parallel_config = SimpleNamespace(
        prefill_context_parallel_size=1,
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
        values.pop("source", None)
        values.update(changes)
        return SimpleNamespace(source=config, **values)

    with patch.object(
        speculator_module,
        "replace",
        side_effect=fake_replace,
    ):
        execution_config = AscendMTPSpeculator._create_draft_execution_config(
            target_config
        )
        speculator = object.__new__(AscendMTPSpeculator)
        speculator.vllm_config = execution_config
        speculator.draft_model_config = draft_model_config
        draft_config = speculator._create_draft_vllm_config()

    assert execution_config.source is target_config
    assert execution_config.parallel_config.source is draft_parallel_config
    assert execution_config.parallel_config.rank == target_parallel_config.rank
    assert execution_config.parallel_config.prefill_context_parallel_size == 1
    assert not execution_config.parallel_config.enable_expert_parallel
    assert draft_config.source is execution_config
    assert draft_config.model_config is draft_model_config
    assert draft_config.parallel_config is execution_config.parallel_config


def test_without_target_pcp_manager_restores_after_error() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.pcp_manager = MagicMock()
    speculator.model_state = SimpleNamespace(pcp_manager=speculator.pcp_manager)

    with pytest.raises(RuntimeError, match="proposal failed"), speculator._without_target_pcp_manager():
        assert speculator.model_state.pcp_manager is None
        raise RuntimeError("proposal failed")

    assert speculator.model_state.pcp_manager is speculator.pcp_manager


def test_set_attn_disables_pcp_only_for_draft_gqa_metadata() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.vllm_config = MagicMock()
    speculator.draft_attn_layer_names = set()

    draft_builder = speculator_module.AscendAttentionPCPMetadataBuilder.__new__(
        speculator_module.AscendAttentionPCPMetadataBuilder
    )
    draft_builder.set_pcp_enabled(True)
    target_builder = speculator_module.AscendAttentionPCPMetadataBuilder.__new__(
        speculator_module.AscendAttentionPCPMetadataBuilder
    )
    target_builder.set_pcp_enabled(True)
    draft_group = SimpleNamespace(
        metadata_builders=[draft_builder],
        backend=speculator_module.AscendAttentionBackend,
    )
    target_attn_groups = [[SimpleNamespace(metadata_builders=[target_builder])]]
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = []

    def set_draft_groups(*_args, **_kwargs) -> None:
        speculator.attn_groups = [[draft_group]]

    with (
        patch.object(
            speculator_module.AutoRegressiveSpeculator,
            "set_attn",
            side_effect=set_draft_groups,
        ),
        patch.object(speculator_module, "set_current_vllm_config", return_value=MagicMock()),
        patch.object(
            speculator_module,
            "_get_graph_update_backend",
            return_value=speculator_module.AscendAttentionBackend,
        ),
    ):
        speculator.set_attn(
            MagicMock(), kv_cache_config, MagicMock(), MagicMock(), target_attn_groups
        )

    assert draft_builder.metadata_cls is AscendMetadata
    assert target_builder.metadata_cls is AscendAttentionPCPMetadata
    assert speculator.attn_architecture == "GQA"


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
    speculator.model_state = SimpleNamespace(
        pcp_manager=speculator.pcp_manager,
    )

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
    speculator.pcp_manager.prepare_replicated_draft_attn.return_value = (
        global_attn_metadata,
        global_slot_mappings,
    )

    with patch.object(parent_cls, "propose", return_value=MagicMock()) as propose:
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

    speculator.pcp_manager.prepare_replicated_draft_attn.assert_called_once_with(
        speculator, input_batch
    )

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
    assert speculator.model_state.pcp_manager is speculator.pcp_manager
