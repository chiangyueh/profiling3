/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_grad_common.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_COMMON_H
#define FLASH_ATTENTION_SCORE_GRAD_COMMON_H
 
#include "common.h"
#include "../../../common/op_kernel/buffers_policy.h"
using namespace fa_base_matmul;
 
constexpr uint8_t SYNC_C1_TO_V2_FLAG[2] = {0, 1};
constexpr uint8_t SYNC_C2_TO_V2_FLAG[2] = {2, 3};
constexpr uint8_t SYNC_V3_TO_C3_FLAG = 4;
constexpr uint8_t SYNC_V4_TO_C5_FLAG = 5;
constexpr uint8_t SYNC_C3_TO_V5_FLAG = 6;
constexpr uint8_t SYNC_C4_TO_V6_FLAG = 7;
constexpr uint8_t SYNC_C4_TO_V3_FLAG = 8;
constexpr uint8_t SYNC_DETER_FIX_FLAG = 9;
constexpr uint8_t SYNC_C5_TO_V4_FLAG = 10;
 
// MM_IDX
constexpr uint8_t DQ_IDX = 0;
constexpr uint8_t DK_IDX = 1;
constexpr uint8_t DV_IDX = 2;
 
template <typename T, bool IS_WRITE_UB>
struct DqkvResPos {
    using PosType = typename std::conditional<IS_WRITE_UB, LocalTensor<T> &, GlobalTensor<T> &>::type;
};
 
 
template <typename T1>
__aicore__ constexpr bool GET_IS_L1_PRELOAD(const uint32_t HEAD_DIM_ALIGN, const uint32_t SPLIT_AXIS,
                                        const bool IS_DETER_OLD, const bool IS_TND, const bool FP8_OPEN_TSCM,
                                        const bool IS_ROPE)
{
    return (HEAD_DIM_ALIGN <= static_cast<uint32_t>(DTemplateType::Aligned192) &&
            ((IS_DETER_OLD == 0) || (IS_DETER_OLD == 1 && IS_ROPE)) && (!IsSameType<T1, float>::value) &&
            (!IsSameType<T1, fp8_e5m2_t>::value) && (!IsSameType<T1, fp8_e4m3fn_t>::value)) ||
           (FP8_OPEN_TSCM && (HEAD_DIM_ALIGN <= static_cast<uint32_t>(DTemplateType::Aligned256)) && (IsSameType<T1, fp8_e5m2_t>::value || IsSameType<T1, fp8_e4m3fn_t>::value));
}
 
template <typename T1>
__aicore__ constexpr bool GET_IS_L1_REUSE(const uint32_t HEAD_DIM_ALIGN, const bool IS_DETER_OLD, const bool FP8_OPEN_TSCM)
{
    return (((!IS_DETER_OLD && HEAD_DIM_ALIGN <= static_cast<uint32_t>(DTemplateType::Aligned256)) ||
             (IS_DETER_OLD && HEAD_DIM_ALIGN <= static_cast<uint32_t>(DTemplateType::Aligned192))) &&
            (!IsSameType<T1, float>::value && !IsSameType<T1, fp8_e5m2_t>::value &&
             !IsSameType<T1, fp8_e4m3fn_t>::value)) ||
           (FP8_OPEN_TSCM && (HEAD_DIM_ALIGN <= static_cast<uint32_t>(DTemplateType::Aligned256)) && (IsSameType<T1, fp8_e5m2_t>::value || IsSameType<T1, fp8_e4m3fn_t>::value));
}
 
// max(mm1, mm2, mm3) + mm4 + mm5
#define IS_DKV_RESIDENT_L0C(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN)                                                    \
    (((CUBE_BASEM) * (HEAD_DIM_ALIGN) * sizeof(float)) + ((CUBE_BASEN) * (HEAD_DIM_ALIGN) * sizeof(float)) +                   \
     ((CUBE_BASEN) > (HEAD_DIM_ALIGN) ? (CUBE_BASEM) * (CUBE_BASEN) * sizeof(float) :                                          \
                                    (CUBE_BASEM) * (HEAD_DIM_ALIGN) * sizeof(float))) <= L0C_MAX_SIZE

#define FagTilingType                                                                                                  \
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<NEED_DETER_PREFIX(DETER_SPARSE_TYPE, IS_TND), IS_TND> \
        *__restrict

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X)                                                                               \
    X(INPUT_TYPE)                                                                                                      \
    X(CALC_TYPE)                                                                                                       \
    X(OUTDTYPE)
 
#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X)                                                                              \
    X(IS_ATTEN_MASK, bool, false)                                                                                      \
    X(IS_PSE, bool, false)                                                                                             \
    X(IS_DROP, bool, false)                                                                                            \
    X(IS_TND, bool, false)                                                                                             \
    X(IS_BN2_MULTIBLK, bool, false)                                                                                           \
    X(DETER_SPARSE_TYPE, uint8_t, 0)                                                                                   \
    X(IS_N_EQUAL, bool, false)                                                                                         \
    X(IS_D_NO_EQUAL, bool, false)                                                                                      \
    X(IS_ROPE, bool, false)                                                                                            \
    X(FP8_OPEN_TSCM, bool, false)                                                                                      \
    X(SPLIT_AXIS, uint8_t, 0)                                                                                          \
    X(s1TemplateType, S1TemplateType, S1TemplateType::Aligned128)                                                      \
    X(s2TemplateType, S2TemplateType, S2TemplateType::Aligned128)                                                      \
    X(dTemplateType, DTemplateType, DTemplateType::Aligned128)
 
/* 1. 生成带默认值的模版Template */
#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type (name) = (default_val),
#define TEMPLATES_DEF                                                                                                  \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = \
                  true>
 
/* 2. 生成不带带默认值的模版Template */
#define GEN_TEMPLATE_TYPE_NODEF(name) typename name,
#define GEN_TEMPLATE_CONST_NODEF(name, type, default_val) type name,
#define TEMPLATES_DEF_NO_DEFAULT                                                                                       \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF)                                                   \
                  CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>
 
/* 3. 生成有默认值, 不带ChildClass的Args */
#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS                                                                                                  \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME)                                                                        \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end
 
/* 4. 生成BASE的有默认值的Template, BASE带ChildClass*/
#define TEMPLATES_DEF_BASE                                                                                             \
    template <typename ChildClass, CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM)                                       \
                                       CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>
 
/* 5. 生成BASE的没有默认值的Template, BASE带ChildClass */
#define TEMPLATES_DEF_BASE_NO_DEFAULT                                                                                  \
    template <typename ChildClass, CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF)                              \
                                       CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>
 
/* 6. 生成BASE的BaseArgs, BASE带ChildClass */
#define TEMPLATE_BASE_ARGS                                                                                             \
    ChildClass, CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end
 
 
#endif // FLASH_ATTENTION_SCORE_GRAD_COMMON_H