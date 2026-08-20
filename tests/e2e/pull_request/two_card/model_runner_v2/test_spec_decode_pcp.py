# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""MRV2 PCP smoke coverage for autoregressive speculative decoding."""

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

SHARED_PREFIX = "This is a shared context for prefix-cache validation. " * 64
PROMPTS = [
    SHARED_PREFIX + "Hello, my name is",
    SHARED_PREFIX + "The president of the United States is",
]


@pytest.mark.parametrize(
    ("model", "speculative_config"),
    [
        pytest.param(
            "wemaster/deepseek_mtp_main_random_bf16",
            {
                "method": "mtp",
                "num_speculative_tokens": 3,
            },
            id="mtp-mla",
        ),
        pytest.param(
            "Qwen/Qwen3-8B",
            {
                "method": "eagle3",
                "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
                "num_speculative_tokens": 3,
            },
            id="eagle3-gqa",
        ),
    ],
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "HCCL_BUFFSIZE": "1024",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_spec_decode_with_pcp(
    model: str,
    speculative_config: dict,
) -> None:
    sampling_params = SamplingParams(max_tokens=16, temperature=0.0)
    with VllmRunner(
        model,
        tensor_parallel_size=1,
        prefill_context_parallel_size=2,
        max_model_len=1024,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        disable_log_stats=False,
        distributed_executor_backend="mp",
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        speculative_config=speculative_config,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4, 8],
        },
    ) as runner:
        uncached_outputs = runner.model.generate(PROMPTS, sampling_params)
        cached_outputs = runner.model.generate(PROMPTS, sampling_params)
        metrics = runner.model.get_metrics()
        num_drafts = sum(metric.value for metric in metrics if metric.name == "vllm:spec_decode_num_drafts")

    uncached_token_ids = [output.outputs[0].token_ids for output in uncached_outputs]
    cached_token_ids = [output.outputs[0].token_ids for output in cached_outputs]
    assert len(uncached_token_ids) == len(PROMPTS)
    assert all(uncached_token_ids)
    assert cached_token_ids == uncached_token_ids
    assert num_drafts > 0
