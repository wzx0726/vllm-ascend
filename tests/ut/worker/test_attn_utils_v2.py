# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata


class _OneTokenPrefillBuilder:
    def build(self, common_prefix_len, common_attn_metadata):
        assert common_prefix_len == 0
        return split_decodes_and_prefills(
            common_attn_metadata,
            treat_short_extends_as_decodes=False,
        )


def test_build_attn_metadata_keeps_one_token_prefill_phase():
    builder = _OneTokenPrefillBuilder()
    attn_group = SimpleNamespace(
        layer_names=["layer.0"],
        get_metadata_builder=lambda _: builder,
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())],
    )
    is_prefilling = torch.tensor([True])

    metadata = build_attn_metadata(
        attn_groups=[[attn_group]],
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_seq_len=1,
        block_tables=(torch.zeros((1, 1), dtype=torch.int32),),
        slot_mappings=(torch.zeros(1, dtype=torch.int64),),
        kv_cache_config=kv_cache_config,
        is_prefilling=is_prefilling,
        seq_lens_np=np.array([1], dtype=np.int32),
        positions=torch.tensor([0], dtype=torch.int64),
    )

    assert metadata["layer.0"] == (
        0,  # num_decodes
        1,  # num_prefills
        0,  # num_decode_tokens
        1,  # num_prefill_tokens
    )
