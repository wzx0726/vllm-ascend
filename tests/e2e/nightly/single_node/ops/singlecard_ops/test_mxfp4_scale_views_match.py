# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import enable_custom_op


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="MegaMoe MXFP4 scales require Ascend A5")
@pytest.mark.parametrize("source_dtype", [torch.uint8, torch.float8_e8m0fnu])
@torch.inference_mode()
def test_mxfp4_scale_views_match_current_experts(source_dtype):
    enable_custom_op()
    matches = torch.ops._C_ascend._mxfp4_scale_views_match

    def scale(value):
        return torch.full((4, 3, 2), value, dtype=torch.uint8, device="npu:0").view(source_dtype)

    first = [scale(127), scale(128)]
    second = [scale(129), scale(130)]
    first_views = [tensor.view(torch.float8_e8m0fnu) for tensor in first]
    second_views = [tensor.view(torch.float8_e8m0fnu) for tensor in second]
    assert matches(first, second, first_views, second_views)

    first[0].view(torch.uint8).copy_(scale(131).view(torch.uint8))
    assert matches(first, second, first_views, second_views)
    expected = torch.full((4, 3, 2), 131, dtype=torch.uint8)
    assert torch.equal(first_views[0].view(torch.uint8).cpu(), expected)

    # The Python Tensor object stays the same while its storage changes.
    first[0].data = scale(132)
    assert not matches(first, second, first_views, second_views)
    assert torch.equal(first_views[0].view(torch.uint8).cpu(), expected)
    first_views[0] = first[0].view(torch.float8_e8m0fnu)
    assert matches(first, second, first_views, second_views)
    assert torch.equal(first_views[0].view(torch.uint8).cpu(), torch.full_like(expected, 132))

    assert not matches(first, second[::-1], first_views, second_views)
    second[0] = second[0].transpose(0, 1)
    assert not matches(first, second, first_views, second_views)
