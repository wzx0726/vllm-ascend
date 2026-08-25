# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""MRV2 PCP smoke coverage for autoregressive speculative decoding."""

import os
from unittest.mock import patch

import pytest
from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from vllm import SamplingParams

LONG_CONTEXT = "This is a long context for PCP prefill validation. " * 64
PROMPTS = [
    LONG_CONTEXT + "Hello, my name is",
    LONG_CONTEXT + "The president of the United States is",
]
MTP_MODEL = os.getenv(
    "SPEC_DECODE_PCP_MTP_MODEL",
    "wemaster/deepseek_mtp_main_random_bf16",
)
EAGLE_TARGET_MODEL = os.getenv(
    "SPEC_DECODE_PCP_EAGLE_TARGET_MODEL",
    "Qwen/Qwen3-8B",
)
EAGLE_DRAFT_MODEL = os.getenv(
    "SPEC_DECODE_PCP_EAGLE_DRAFT_MODEL",
    "RedHatAI/Qwen3-8B-speculator.eagle3",
)


@pytest.mark.parametrize(
    ("model", "speculative_config"),
    [
        pytest.param(
            MTP_MODEL,
            {
                "method": "mtp",
                "num_speculative_tokens": 3,
            },
            id="mtp-mla",
        ),
        pytest.param(
            EAGLE_TARGET_MODEL,
            {
                "method": "eagle3",
                "model": EAGLE_DRAFT_MODEL,
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
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "ASCEND_RT_VISIBLE_DEVICES": "0,1",
        "HCCL_BUFFSIZE": "1024",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_spec_decode_with_pcp(
    model: str,
    speculative_config: dict,
) -> None:
    sampling_params = SamplingParams(max_tokens=16, temperature=0.0)
    runner_kwargs = {
        "tensor_parallel_size": 1,
        "prefill_context_parallel_size": 2,
        "max_model_len": 1024,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 4,
        "disable_log_stats": False,
        "distributed_executor_backend": "mp",
        "enable_chunked_prefill": True,
        "compilation_config": {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4, 8],
        },
    }
    with VllmRunner(
        model,
        speculative_config=speculative_config,
        **runner_kwargs,
    ) as runner:
        outputs = runner.model.generate(PROMPTS, sampling_params)
        metrics = runner.model.get_metrics()
        num_drafts = sum(metric.value for metric in metrics if metric.name == "vllm:spec_decode_num_drafts")

    token_ids = [output.outputs[0].token_ids for output in outputs]
    assert len(token_ids) == len(PROMPTS)
    assert all(token_ids)
    assert num_drafts > 0
