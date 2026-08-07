# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendC8AttentionBackendImpl,
    AscendMetadata,
)
from vllm_ascend.attention.context_parallel.attention_cp import (
    AscendAttentionDCPImpl,
    AscendAttentionDCPMetadata,
    AscendAttentionDCPMetadataBuilder,
    AscendAttentionPCPImpl,
    AscendAttentionPCPMetadata,
    AscendAttentionPCPMetadataBuilder,
    AscendMetadataForDecode,
)
from vllm_ascend.attention.context_parallel.common_cp import (
    _update_out_and_lse,
)


def test_gqa_dcp_extends_v1_backend_without_polluting_base_metadata() -> None:
    assert issubclass(AscendAttentionDCPImpl, AscendAttentionBackendImpl)
    assert issubclass(
        AscendAttentionDCPMetadataBuilder,
        AscendAttentionMetadataBuilder,
    )
    assert AscendAttentionDCPMetadataBuilder.metadata_cls is (AscendAttentionDCPMetadata)
    assert not hasattr(AscendMetadata(), "decode_meta")
    assert not hasattr(AscendMetadata(), "prefill")


def test_gqa_pcp_extends_v1_backend_without_polluting_base_metadata() -> None:
    assert issubclass(AscendAttentionPCPImpl, AscendAttentionBackendImpl)
    assert issubclass(
        AscendAttentionPCPMetadataBuilder,
        AscendAttentionMetadataBuilder,
    )
    assert AscendAttentionPCPMetadataBuilder.metadata_cls is (AscendAttentionPCPMetadata)
    assert not hasattr(AscendMetadata(), "pcp_slot_mapping")


def test_pcp_builder_selects_rank_slots_and_uses_cached_prefill() -> None:
    builder = AscendAttentionPCPMetadataBuilder.__new__(AscendAttentionPCPMetadataBuilder)
    builder.pcp_size = 2
    builder.pcp_rank = 1
    common_metadata = SimpleNamespace(
        slot_mapping=torch.tensor(
            [100, 101, 102, 103, -1, -1, 201, 202, 203, -1],
            dtype=torch.int64,
        )
    )
    base_metadata = AscendAttentionPCPMetadata(
        num_actual_tokens=4,
        num_decode_tokens=1,
        num_prefills=1,
        attn_state=AscendAttentionState.PrefillNoCache,
    )

    with patch.object(
        AscendAttentionMetadataBuilder,
        "build",
        return_value=base_metadata,
    ):
        metadata = builder.build(0, common_metadata)

    assert torch.equal(
        metadata.slot_mapping,
        torch.tensor([100, 201, 202, 203], dtype=torch.int64),
    )
    assert metadata.pcp_slot_mapping is common_metadata.slot_mapping
    assert metadata.pcp_local_num_input_tokens == 5
    assert metadata.attn_state == AscendAttentionState.ChunkedPrefill


@pytest.mark.parametrize(
    "initial_state",
    [
        AscendAttentionState.PrefillNoCache,
        AscendAttentionState.PrefillCacheHit,
        AscendAttentionState.DecodeOnly,
        AscendAttentionState.ChunkedPrefill,
    ],
)
def test_pcp_builder_uses_chunked_state_for_every_prefill(initial_state) -> None:
    builder = AscendAttentionPCPMetadataBuilder.__new__(AscendAttentionPCPMetadataBuilder)
    builder.pcp_size = 2
    builder.pcp_rank = 0
    common_metadata = SimpleNamespace(slot_mapping=torch.tensor([10, 11, -1, 20, 21, 22], dtype=torch.int64))
    base_metadata = AscendAttentionPCPMetadata(
        num_actual_tokens=2,
        num_decode_tokens=0,
        num_prefills=1,
        attn_state=initial_state,
    )

    with patch.object(
        AscendAttentionMetadataBuilder,
        "build",
        return_value=base_metadata,
    ):
        metadata = builder.build(0, common_metadata)

    assert torch.equal(metadata.slot_mapping, torch.tensor([10, 11]))
    assert metadata.pcp_local_num_input_tokens == 3
    assert metadata.attn_state == AscendAttentionState.ChunkedPrefill


def test_pcp_builder_preserves_decode_only_state() -> None:
    builder = AscendAttentionPCPMetadataBuilder.__new__(AscendAttentionPCPMetadataBuilder)
    builder.pcp_size = 2
    builder.pcp_rank = 1
    common_metadata = SimpleNamespace(slot_mapping=torch.tensor([10, 11, -1, -1], dtype=torch.int64))
    base_metadata = AscendAttentionPCPMetadata(
        num_actual_tokens=2,
        num_decode_tokens=2,
        num_prefills=0,
        attn_state=AscendAttentionState.DecodeOnly,
    )

    with patch.object(
        AscendAttentionMetadataBuilder,
        "build",
        return_value=base_metadata,
    ):
        metadata = builder.build(0, common_metadata)

    assert torch.equal(metadata.slot_mapping, torch.tensor([10, 11]))
    assert metadata.attn_state == AscendAttentionState.DecodeOnly


def test_pcp_builder_rejects_nondivisible_expanded_slots() -> None:
    builder = AscendAttentionPCPMetadataBuilder.__new__(AscendAttentionPCPMetadataBuilder)
    builder.pcp_size = 2
    builder.pcp_rank = 0
    common_metadata = SimpleNamespace(slot_mapping=torch.arange(9))
    base_metadata = AscendAttentionPCPMetadata(num_actual_tokens=4)

    with (
        patch.object(
            AscendAttentionMetadataBuilder,
            "build",
            return_value=base_metadata,
        ),
        np.testing.assert_raises_regex(RuntimeError, "must be divisible"),
    ):
        builder.build(0, common_metadata)


def _make_pcp_impl() -> AscendAttentionPCPImpl:
    impl = AscendAttentionPCPImpl.__new__(AscendAttentionPCPImpl)
    impl.key_cache = None
    impl.value_cache = None
    impl.kv_sharing_target_layer_name = None
    impl.is_kv_producer = True
    return impl


def test_pcp_cache_gather_keeps_one_decode_and_all_padded_prefills() -> None:
    impl = _make_pcp_impl()
    query = torch.zeros((4, 2, 1))
    key = torch.tensor([0, 1, 2, 0]).reshape(4, 1, 1)
    value = torch.tensor([100, 101, 102, 100]).reshape(4, 1, 1)
    key_cache = torch.empty((8, 1, 1))
    value_cache = torch.empty((8, 1, 1))
    output = torch.empty((4, 2, 1))
    metadata = AscendAttentionPCPMetadata(
        num_decode_tokens=1,
        pcp_local_num_input_tokens=4,
        pcp_slot_mapping=torch.tensor(
            [10, 11, 12, -1, -1, 21, 22, -1],
            dtype=torch.int64,
        ),
    )

    def all_gather(tensor, dim):
        return torch.cat((tensor, tensor + 10), dim=dim)

    pcp_group = SimpleNamespace(world_size=2, all_gather=all_gather)
    with (
        patch(
            "vllm.model_executor.layers.attention.pcp.get_pcp_group",
            return_value=pcp_group,
        ),
        patch(
            "vllm_ascend.attention.context_parallel.attention_cp.DeviceOperator.reshape_and_cache"
        ) as reshape_and_cache,
        patch("vllm_ascend.attention.context_parallel.attention_cp.notify_kv_cache_written") as notify_cache_written,
    ):
        result = impl.reshape_and_cache(
            query,
            key,
            value,
            (key_cache, value_cache),
            metadata,
            output,
        )

    kwargs = reshape_and_cache.call_args.kwargs
    assert torch.equal(
        kwargs["key"].flatten(),
        torch.tensor([0, 1, 2, 0, 11, 12, 10]),
    )
    assert torch.equal(
        kwargs["value"].flatten(),
        torch.tensor([100, 101, 102, 100, 111, 112, 110]),
    )
    assert torch.equal(
        kwargs["slot_mapping"],
        torch.tensor([10, 11, 12, -1, 21, 22, -1]),
    )
    assert kwargs["key_cache"] is key_cache
    assert kwargs["value_cache"] is value_cache
    notify_cache_written.assert_called_once_with()
    assert result[0] is query
    assert result[1] is key
    assert result[2] is value
    assert result[3] is output


def test_pcp_cache_gather_pure_prefill_odd_length_with_padding() -> None:
    impl = _make_pcp_impl()
    query = torch.zeros((3, 2, 1))
    key = torch.tensor([0, 1, 0]).reshape(3, 1, 1)
    value = torch.tensor([100, 101, 100]).reshape(3, 1, 1)
    key_cache = torch.empty((8, 1, 1))
    value_cache = torch.empty((8, 1, 1))
    output = torch.empty((3, 2, 1))
    metadata = AscendAttentionPCPMetadata(
        num_decode_tokens=0,
        pcp_local_num_input_tokens=3,
        pcp_slot_mapping=torch.tensor([10, 11, -1, 12, 13, 14]),
    )

    def all_gather(tensor, dim):
        remote_delta = tensor.new_tensor([2, 2, 4]).reshape_as(tensor)
        return torch.cat((tensor, tensor + remote_delta), dim=dim)

    pcp_group = SimpleNamespace(world_size=2, all_gather=all_gather)
    with (
        patch(
            "vllm.model_executor.layers.attention.pcp.get_pcp_group",
            return_value=pcp_group,
        ),
        patch(
            "vllm_ascend.attention.context_parallel.attention_cp.DeviceOperator.reshape_and_cache"
        ) as reshape_and_cache,
        patch("vllm_ascend.attention.context_parallel.attention_cp.notify_kv_cache_written") as notify_cache_written,
    ):
        impl.reshape_and_cache(
            query,
            key,
            value,
            (
                key_cache,
                value_cache,
            ),
            metadata,
            output,
        )

    kwargs = reshape_and_cache.call_args.kwargs
    assert torch.equal(
        kwargs["key"].flatten(),
        torch.tensor([0, 1, 0, 2, 3, 4]),
    )
    assert torch.equal(
        kwargs["value"].flatten(),
        torch.tensor([100, 101, 100, 102, 103, 104]),
    )
    assert torch.equal(
        kwargs["slot_mapping"],
        torch.tensor([10, 11, -1, 12, 13, 14]),
    )
    notify_cache_written.assert_called_once_with()


def test_pcp_cache_hit_c8_chunked_prefill_gathers_and_dequantizes() -> None:
    impl = AscendC8AttentionBackendImpl.__new__(AscendC8AttentionBackendImpl)
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.head_size = 4
    impl.scale = 1.0
    impl.key_cache = torch.empty((2, 32, 1, 4), dtype=torch.int8)
    impl.value_cache = torch.empty((2, 32, 1, 4), dtype=torch.int8)

    query = torch.zeros((1, 2, 4))
    new_key = torch.zeros((1, 1, 4))
    new_value = torch.zeros((1, 1, 4))
    output = torch.empty_like(query)
    metadata = AscendAttentionPCPMetadata(
        num_decode_tokens=0,
        num_decodes=0,
        num_prefills=1,
        actual_seq_lengths_q=[1],
        seq_lens_list=[33],
        block_tables=torch.tensor([[0, 1]], dtype=torch.long),
        attn_mask=None,
    )
    dense_key = torch.zeros((33, 1, 4))
    dense_value = torch.zeros((33, 1, 4))
    layer = SimpleNamespace()

    with (
        patch.object(
            impl,
            "_nz_5d_view",
            side_effect=[impl.key_cache, impl.value_cache],
        ),
        patch.object(
            impl,
            "_dequant_paged_kv_to_dense",
            return_value=(dense_key, dense_value),
        ) as dequant,
        patch(
            "vllm_ascend.attention.attention_v1.torch_npu.npu_fused_infer_attention_score",
            return_value=(torch.ones_like(query), None),
        ) as fused_attention,
    ):
        result = impl._forward_c8_chunked_prefill(
            query,
            new_key,
            new_value,
            metadata,
            output,
            layer,
        )

    dequant.assert_called_once()
    assert dequant.call_args.args[2] is metadata.block_tables
    assert dequant.call_args.args[3] == [33]
    fused_kwargs = fused_attention.call_args.kwargs
    assert fused_kwargs["key"] is dense_key
    assert fused_kwargs["value"] is dense_value
    assert fused_kwargs["actual_seq_lengths"] == [1]
    torch.testing.assert_close(result, torch.ones_like(query))


def test_pcp_decode_only_does_not_all_gather_kv() -> None:
    impl = _make_pcp_impl()
    query = torch.zeros((2, 2, 1))
    key = torch.tensor([1, 2]).reshape(2, 1, 1)
    value = torch.tensor([11, 12]).reshape(2, 1, 1)
    output = torch.empty((2, 2, 1))
    metadata = AscendAttentionPCPMetadata(
        num_decode_tokens=2,
        pcp_local_num_input_tokens=2,
        pcp_slot_mapping=torch.tensor([10, 11, -1, -1]),
    )

    with (
        patch("vllm.model_executor.layers.attention.pcp.get_pcp_group") as get_pcp_group,
        patch(
            "vllm_ascend.attention.context_parallel.attention_cp.DeviceOperator.reshape_and_cache"
        ) as reshape_and_cache,
        patch("vllm_ascend.attention.context_parallel.attention_cp.notify_kv_cache_written"),
    ):
        impl.reshape_and_cache(
            query,
            key,
            value,
            (torch.empty((4, 1, 1)), torch.empty((4, 1, 1))),
            metadata,
            output,
        )

    get_pcp_group.assert_not_called()
    kwargs = reshape_and_cache.call_args.kwargs
    assert torch.equal(kwargs["key"], key)
    assert torch.equal(kwargs["value"], value)
    assert torch.equal(kwargs["slot_mapping"], torch.tensor([10, 11]))


def test_dcp_chunked_request_mask_marks_nonempty_contexts() -> None:
    local_context_lens = torch.tensor(
        [
            [0, 0],
            [4, 0],
            [0, 7],
        ],
        dtype=torch.int32,
    )

    assert AscendAttentionDCPMetadataBuilder._get_chunked_req_mask(local_context_lens) == [
        False,
        True,
        True,
    ]


def test_dcp_decode_metadata_keeps_rank_local_context_lengths() -> None:
    local_context_lens = np.array([[11, 12], [21, 22]], dtype=np.int32)
    block_tables = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    metadata = AscendMetadataForDecode(
        num_computed_tokens_of_dcp=local_context_lens,
        block_tables=block_tables,
    )

    np.testing.assert_array_equal(metadata.num_computed_tokens_of_dcp[:, 1], [12, 22])
    assert metadata.block_tables is block_tables


def test_dcp_partial_attention_merge_matches_weighted_reference() -> None:
    outputs = torch.tensor(
        [
            [[[[1.0, 3.0]]]],
            [[[[5.0, 7.0]]]],
        ]
    ).reshape(2, 1, 1, 2)
    lse = torch.tensor([0.0, np.log(3.0)], dtype=torch.float32).reshape(2, 1, 1, 1)

    output, merged_lse = _update_out_and_lse(outputs, lse)

    torch.testing.assert_close(output, torch.tensor([[[4.0, 6.0]]]))
    torch.testing.assert_close(merged_lse, torch.tensor([[[np.log(4.0)]]], dtype=torch.float32))


def test_pcp_builder_keeps_short_extend_in_prefill() -> None:
    builder = AscendAttentionPCPMetadataBuilder.__new__(
        AscendAttentionPCPMetadataBuilder
    )
    builder.decode_threshold = 1
    common_metadata = SimpleNamespace(
        context_parallel_metadata=None,
        max_query_len=4,
        num_reqs=2,
        num_actual_tokens=5,
        query_start_loc_cpu=torch.tensor([0, 1, 5], dtype=torch.int32),
        is_prefilling=torch.tensor([True, True], dtype=torch.bool),
    )

    result = builder._split_decodes_and_prefills(common_metadata)

    assert result == (0, 2, 0, 5)

