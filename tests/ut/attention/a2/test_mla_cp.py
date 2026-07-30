# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.attention.context_parallel.mla_cp import (
    AscendMLADCPDecodeMetadata,
    AscendMlaDCPImpl,
    AscendMlaDCPMetadataBuilder,
    AscendMLAPCPImpl,
    AscendMLAPCPMetadata,
    AscendMLAPCPMetadataBuilder,
    DCPChunkedContextMetadata,
)
from vllm_ascend.attention.mla_v1 import (
    AscendMLADecodeMetadata,
    AscendMLAImpl,
    AscendMLAMetadata,
    AscendMLAMetadataBuilder,
    AscendMLAPrefillMetadata,
)


def _make_pcp_metadata(
    *,
    num_actual_tokens: int,
    num_decode_tokens: int,
) -> AscendMLAPCPMetadata:
    return AscendMLAPCPMetadata(
        num_actual_tokens=num_actual_tokens,
        slot_mapping=torch.empty(0, dtype=torch.int64),
        query_start_loc=torch.tensor([0, num_actual_tokens], dtype=torch.int32),
        seq_lens=torch.tensor([num_actual_tokens], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([num_actual_tokens], dtype=torch.int32),
        block_tables=torch.zeros(1, 1, dtype=torch.int32),
        num_decodes=int(num_decode_tokens > 0),
        num_decode_tokens=num_decode_tokens,
        num_prefills=int(num_actual_tokens > num_decode_tokens),
    )


def test_mla_pcp_metadata_selects_rank_local_prefill_slots() -> None:
    expanded_slots = torch.tensor(
        [
            10,
            11,
            12,
            -1,
            20,
            21,
            -1,
            -1,
        ],
        dtype=torch.int64,
    )
    common_metadata = SimpleNamespace(slot_mapping=expanded_slots)

    for pcp_rank, num_actual_tokens, expected in (
        (0, 3, [10, 11, 12]),
        (1, 2, [20, 21]),
    ):
        metadata = _make_pcp_metadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=0,
        )
        builder = AscendMLAPCPMetadataBuilder.__new__(AscendMLAPCPMetadataBuilder)
        builder.pcp_size = 2
        builder.pcp_rank = pcp_rank
        with patch.object(
            AscendMLAMetadataBuilder,
            "build",
            return_value=metadata,
        ):
            result = builder.build(0, common_metadata)

        torch.testing.assert_close(
            result.slot_mapping,
            torch.tensor(expected, dtype=torch.int64),
        )
        assert result.pcp_slot_mapping is expanded_slots
        assert result.pcp_local_num_input_tokens == 4


def test_mla_pcp_metadata_uses_rank_zero_decode_slots() -> None:
    expanded_slots = torch.tensor(
        [
            5,
            10,
            11,
            -1,
            -1,
            20,
            21,
            -1,
        ],
        dtype=torch.int64,
    )
    common_metadata = SimpleNamespace(slot_mapping=expanded_slots)

    for pcp_rank, expected in ((0, [5, 10, 11]), (1, [5, 20, 21])):
        metadata = _make_pcp_metadata(
            num_actual_tokens=3,
            num_decode_tokens=1,
        )
        builder = AscendMLAPCPMetadataBuilder.__new__(AscendMLAPCPMetadataBuilder)
        builder.pcp_size = 2
        builder.pcp_rank = pcp_rank
        with patch.object(
            AscendMLAMetadataBuilder,
            "build",
            return_value=metadata,
        ):
            result = builder.build(0, common_metadata)

        torch.testing.assert_close(
            result.slot_mapping,
            torch.tensor(expected, dtype=torch.int64),
        )


def test_mla_pcp_decode_metadata_uses_persistent_rank_zero_view() -> None:
    expanded_slots = torch.tensor(
        [5, 6, -1, -1, 50, 60, -1, -1],
        dtype=torch.int64,
    )
    common_metadata = SimpleNamespace(slot_mapping=expanded_slots)
    metadata = _make_pcp_metadata(
        num_actual_tokens=4,
        num_decode_tokens=4,
    )
    metadata.num_decodes = 4
    metadata.query_start_loc = torch.arange(5, dtype=torch.int32)

    builder = AscendMLAPCPMetadataBuilder.__new__(AscendMLAPCPMetadataBuilder)
    builder.pcp_size = 2
    builder.pcp_rank = 1
    with patch.object(
        AscendMLAMetadataBuilder,
        "build",
        return_value=metadata,
    ):
        result = builder.build(0, common_metadata)

    assert result.num_actual_tokens == 4
    assert result.num_decodes == 4
    assert result.num_prefills == 0
    assert result.slot_mapping.data_ptr() == expanded_slots.data_ptr()
    assert result.slot_mapping.tolist() == [5, 6, -1, -1]

    expanded_slots[1] = 99
    assert result.slot_mapping[1].item() == 99


def test_mla_pcp_prefill_gathers_cache_inputs_and_keeps_local_kv() -> None:
    impl = AscendMLAPCPImpl.__new__(AscendMLAPCPImpl)
    impl.num_heads = 1
    impl.num_kv_heads = 1
    impl.qk_head_dim = 3
    impl.qk_nope_head_dim = 2
    impl.v_head_dim = 1
    impl.q_proj = lambda x: (
        torch.arange(x.shape[0] * 3, dtype=x.dtype).view(x.shape[0], 3),
        None,
    )
    impl.rope_single = lambda x, cos, sin: x
    impl.kv_b_proj = lambda x: (torch.cat((x, x[:, :1]), dim=-1), None)

    captured: dict[str, torch.Tensor] = {}

    def fake_gather(tensors, slot_mapping, num_decode_tokens):
        assert num_decode_tokens == 0
        captured["local_kv"] = tensors[0]
        captured["local_cos"] = tensors[1]
        captured["slots"] = slot_mapping
        return (
            tuple(torch.cat((tensor + 100, tensor), dim=0) for tensor in tensors),
            slot_mapping,
        )

    def fake_exec_kv_prefill(kv, cos, sin, kv_cache, slots):
        captured["gathered_kv"] = kv
        captured["cache_slots"] = slots
        return cos, kv[:, :2]

    impl.exec_kv_prefill = fake_exec_kv_prefill
    pcp_group = SimpleNamespace(world_size=2, rank_in_group=1)
    prefill_metadata = AscendMLAPrefillMetadata(
        attn_mask=None,
        query_lens=torch.tensor([2]),
        seq_lens=[3],
        context_lens=torch.tensor([1]),
        input_positions=torch.arange(2),
        query_start_loc=torch.tensor([0, 2]),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        max_query_len=2,
        max_seq_lens=3,
        cos=torch.tensor([[1.0], [2.0]]),
        sin=torch.tensor([[3.0], [4.0]]),
    )
    metadata = _make_pcp_metadata(num_actual_tokens=3, num_decode_tokens=1)
    metadata.prefill = prefill_metadata
    metadata.pcp_slot_mapping = torch.tensor(
        [5, 10, 11, -1, -1, 20, 21, -1],
        dtype=torch.int64,
    )
    metadata.pcp_local_num_input_tokens = 4

    q_c = torch.zeros(4, 1)
    kv_no_split = torch.arange(12, dtype=torch.float32).view(4, 3)
    with (
        patch(
            "vllm_ascend.attention.context_parallel.mla_cp.get_pcp_group",
            return_value=pcp_group,
        ),
        patch(
            "vllm_ascend.attention.context_parallel.mla_cp._gather_prefill_cache_inputs",
            side_effect=fake_gather,
        ),
    ):
        result = impl.mla_preprocess_prefill(
            q_c,
            kv_no_split,
            (torch.empty(0), torch.empty(0)),
            metadata,
        )

    torch.testing.assert_close(
        captured["slots"],
        torch.tensor([10, 11, -1, 20, 21, -1]),
    )
    torch.testing.assert_close(captured["local_kv"], kv_no_split[1:4])
    torch.testing.assert_close(captured["local_cos"][-1], torch.zeros(1))
    assert captured["gathered_kv"].shape[0] == 6
    torch.testing.assert_close(result.k_nope, kv_no_split[1:3, :2].view(2, 1, 2))
    torch.testing.assert_close(result.k_pe, torch.tensor([[[1.0]], [[2.0]]]))
    assert result.q_nope.shape[0] == 2
    assert result.value.shape == (2, 1, 1)


def test_mla_dcp_extends_v1_backend() -> None:
    assert issubclass(AscendMlaDCPImpl, AscendMLAImpl)
    assert issubclass(
        AscendMlaDCPMetadataBuilder,
        AscendMLAMetadataBuilder,
    )
    assert AscendMlaDCPMetadataBuilder.decode_metadata_cls is (AscendMLADCPDecodeMetadata)
    base_fields = {field.name for field in fields(AscendMLADecodeMetadata)}
    dcp_fields = {field.name for field in fields(AscendMLADCPDecodeMetadata)}
    assert {"cp_seq_len", "dcp_mtp_attn_mask"}.isdisjoint(base_fields)
    assert {"cp_seq_len", "dcp_mtp_attn_mask"} <= dcp_fields


def test_mla_dcp_reorg_decode_query_gathers_fused_query() -> None:
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.dcp_size = 2
    impl.kv_lora_rank = 3
    impl.qk_rope_head_dim = 2
    q_nope = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    q_pe = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)

    group = SimpleNamespace(all_gather=lambda tensor, dim: torch.cat([tensor, tensor + 100], dim=dim))
    impl.dcp_group = group
    gathered_nope, gathered_pe = impl.reorg_decode_q(q_nope, q_pe)

    assert gathered_nope.shape == (1, 4, 3)
    assert gathered_pe.shape == (1, 4, 2)
    torch.testing.assert_close(gathered_nope[:, :2], q_nope)
    torch.testing.assert_close(gathered_pe[:, :2], q_pe)
    torch.testing.assert_close(gathered_nope[:, 2:], q_nope + 100)
    torch.testing.assert_close(gathered_pe[:, 2:], q_pe + 100)


def test_mla_dcp_uses_padded_local_chunk_lengths() -> None:
    padded_lengths = torch.tensor([[4, 2], [1, 0]], dtype=torch.int32)
    chunked = DCPChunkedContextMetadata(
        cu_seq_lens=torch.tensor([0, 2]),
        starts=torch.zeros(1, dtype=torch.int32),
        seq_tot=[6, 1],
        max_seq_lens=[4, 1],
        workspace=torch.empty(0),
        chunk_seq_lens=torch.empty(0, dtype=torch.int32),
        chunk_seq_lens_npu=torch.empty(0, dtype=torch.int32),
        chunk_actual_seq_lengths_kv_list=[[4, 6], [1, 1]],
        padded_chunk_seq_lens_npu=padded_lengths,
    )
    metadata = AscendMLAMetadata(
        num_actual_tokens=2,
        slot_mapping=torch.arange(2),
        query_start_loc=torch.tensor([0, 2]),
        seq_lens=torch.tensor([2]),
        seq_lens_cpu=torch.tensor([2]),
        block_tables=torch.zeros(1, 1, dtype=torch.int32),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        prefill=AscendMLAPrefillMetadata(
            attn_mask=None,
            query_lens=torch.tensor([2]),
            seq_lens=[2],
            context_lens=torch.tensor([0]),
            input_positions=torch.arange(2),
            query_start_loc=torch.tensor([0, 2]),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            max_query_len=2,
            max_seq_lens=2,
            chunked_context=chunked,
        ),
    )
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)

    torch.testing.assert_close(impl.get_context_seq_len_npu(1, metadata), padded_lengths[1])
