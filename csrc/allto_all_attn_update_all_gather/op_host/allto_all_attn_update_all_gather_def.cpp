/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather OpDef Registration
 *
 * Fused {AlltoAll + LSE-weighted attention update + head AllGather}, in-place on
 * attn / lse along the CP (context parallel) communication group.
 *
 *   attn_ref [totalT, n_per_cp · D] bf16   (Input & Output, same GM address)
 *   lse_ref  [totalT, n_per_cp]     fp32   (Input & Output, same GM address)
 *   mask_num []                     int32  (mask_per_rank; b0_total = mask_per_rank · cp_size)
 *
 * Active rows [0, b0_total): Phase A peermem-pack → Phase B LSE-weighted reduce
 * → Phase C peermem head-AllGather (reverse stride) back to attn/lse.
 * Inactive rows [b0_total, totalT): inplace pass-through (kernel does not touch;
 * opbuild SetRef guarantees input/output share GM address).
 *
 * Inplace contract: Input("X_ref") + Output("X_ref") same-name + _ref suffix
 * → opbuild emits NnopbaseSetRef.
 */

#include "register/op_def_registry.h"

namespace ops {

class AlltoAllAttnUpdateAllGather : public OpDef {
public:
    explicit AlltoAllAttnUpdateAllGather(const char *name) : OpDef(name) {
        // ===== Inputs =====
        // attn_ref / lse_ref: 同名 Input + Output + _ref 后缀 = inplace SetRef gate
        this->Input("attn_ref")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("lse_ref")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mask_num")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // ===== Outputs (inplace: same name as Inputs, with _ref suffix) =====
        this->Output("attn_ref")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("lse_ref")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // ===== Attributes =====
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("group_size").AttrType(REQUIRED).Int();

        // ===== AICore Configuration =====
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);

        // ===== MC2 Communication Domain Declaration =====
        this->MC2().HcclGroup("group");
    }
};

OP_ADD(AlltoAllAttnUpdateAllGather);

}  // namespace ops
