/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather InferShape & InferDataType
 *
 * Inplace: Input and Output share the same name (attn_ref, lse_ref). Output
 * shapes = input shapes — the kernel only rearranges rows + does LSE-weighted
 * reduce + head-AllGather, so tensor shapes are unchanged.
 */

#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShapeAlltoAllAttnUpdateAllGather(gert::InferShapeContext* context) {
    auto attnShape = context->GetOutputShape(0);    // attn_ref [totalT, n_per_cp · D]
    auto lseShape  = context->GetOutputShape(1);    // lse_ref  [totalT, n_per_cp]

    if (context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
        attnShape == nullptr || lseShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Output shapes match input shapes exactly (in-place, no shape change).
    *attnShape = *context->GetInputShape(0);
    *lseShape  = *context->GetInputShape(1);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAlltoAllAttnUpdateAllGather(gert::InferDataTypeContext* context) {
    // attn_ref: BF16 (same as Input attn_ref)
    context->SetOutputDataType(0, context->GetInputDataType(0));
    // lse_ref:  FLOAT (same as Input lse_ref)
    context->SetOutputDataType(1, context->GetInputDataType(1));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AlltoAllAttnUpdateAllGather)
    .InferShape(InferShapeAlltoAllAttnUpdateAllGather)
    .InferDataType(InferDataTypeAlltoAllAttnUpdateAllGather);

}  // namespace ops
