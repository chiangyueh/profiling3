/**
 * Host-only FlashAttentionScore tiling probe.
 *
 * This follows the same UT path as the upstream operator tests: construct a
 * TilingContextPara, call the registered Host tiling function, and print the
 * encoded result.  It never launches an NPU kernel.
 */
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "../../../../op_host/flash_attention_score_tiling_common.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace {

using Desc = gert::TilingContextPara::TensorDescription;
using Attr = gert::TilingContextPara::OpAttr;

int64_t EnvI64(const char *name, int64_t fallback)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const long long value = std::strtoll(raw, &end, 10);
    if (end == raw || *end != '\0') {
        ADD_FAILURE() << "invalid integer " << name << '=' << raw;
        return fallback;
    }
    return static_cast<int64_t>(value);
}

gert::StorageShape Storage(const std::vector<int64_t> &dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

Desc Tensor(const std::vector<int64_t> &dims, ge::DataType dtype,
            bool isConst = false, void *value = nullptr)
{
    return Desc(Storage(dims), dtype, ge::FORMAT_ND, isConst, value);
}

std::vector<int64_t> DataShape(int64_t layout, int64_t b, int64_t n,
                               int64_t s, int64_t d)
{
    if (layout == 3) {
        return {b * s, n, d};
    }
    return {b, n, s, d};
}

std::vector<int64_t> SoftmaxShape(int64_t layout, int64_t b, int64_t n, int64_t s)
{
    if (layout == 3) {
        return {b * s, n, 8};
    }
    return {b, n, s, 8};
}

const char *LayoutName(int64_t layout)
{
    return layout == 3 ? "TND" : "BNSD";
}

void PrintResult(const TilingInfo &info)
{
    const char *caseName = std::getenv("ATTENTION_CASE_NAME");
    std::cout << "ATTENTION_HOST_TILING"
              << "\tcase=" << (caseName == nullptr ? "unknown" : caseName)
              << "\toperator=flash_attention_score"
              << "\tkey=" << info.tilingKey
              << "\tblock_dim=" << info.blockNum
              << "\tworkspace=";
    for (size_t i = 0; i < info.workspaceSizes.size(); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << info.workspaceSizes[i];
    }
    std::cout << "\tdata_hex=" << std::hex << std::setfill('0');
    for (size_t i = 0; i < info.tilingDataSize; ++i) {
        std::cout << std::setw(2) << static_cast<unsigned int>(info.tilingData[i]);
    }
    std::cout << std::dec << std::endl;
}

TEST(AttentionHostTiling, Forward)
{
    const int64_t b = EnvI64("FA_B", 1);
    const int64_t n1 = EnvI64("FA_N1", 8);
    const int64_t n2 = EnvI64("FA_N2", 1);
    const int64_t s1 = EnvI64("FA_S1", 128);
    const int64_t s2 = EnvI64("FA_S2", 1536);
    const int64_t d = EnvI64("FA_D", 128);
    const int64_t dv = EnvI64("FA_DV", d);
    const int64_t layout = EnvI64("FA_LAYOUT", 0);
    const int64_t dtypeBytes = EnvI64("FA_DTYPE_BYTES", 4);
    const bool hasDrop = EnvI64("FA_HAS_DROP", 0) != 0;

    ASSERT_GT(b, 0);
    ASSERT_GT(n1, 0);
    ASSERT_GT(n2, 0);
    ASSERT_EQ(n1 % n2, 0);
    ASSERT_GT(s1, 0);
    ASSERT_GT(s2, 0);
    ASSERT_GT(d, 0);
    ASSERT_GT(dv, 0);
    ASSERT_TRUE(layout == 0 || layout == 3);
    ASSERT_TRUE(dtypeBytes == 2 || dtypeBytes == 4);

    const ge::DataType dtype = dtypeBytes == 2 ? ge::DT_FLOAT16 : ge::DT_FLOAT;
    std::vector<int64_t> actualQ;
    std::vector<int64_t> actualKv;
    if (layout == 3) {
        for (int64_t index = 1; index <= b; ++index) {
            actualQ.push_back(index * s1);
            actualKv.push_back(index * s2);
        }
    }

    std::vector<Desc> inputs = {
        Tensor(DataShape(layout, b, n1, s1, d), dtype),
        Tensor(DataShape(layout, b, n2, s2, d), dtype),
        Tensor(DataShape(layout, b, n2, s2, dv), dtype),
        Tensor({}, dtype),
        hasDrop ? Tensor({(b * n1 * s1 * s2 + 7) / 8}, ge::DT_UINT8)
                : Tensor({}, ge::DT_UINT8),
        Tensor({}, dtype),
        Tensor({}, ge::DT_UINT8),
        Tensor({}, ge::DT_INT64),
        layout == 3 ? Tensor({b}, ge::DT_INT64, true, actualQ.data())
                    : Tensor({}, ge::DT_INT64),
        layout == 3 ? Tensor({b}, ge::DT_INT64, true, actualKv.data())
                    : Tensor({}, ge::DT_INT64),
        Tensor({}, ge::DT_INT64),
        Tensor({}, ge::DT_INT64),
        Tensor({}, ge::DT_FLOAT),
        Tensor({}, ge::DT_FLOAT),
        Tensor({}, ge::DT_FLOAT),
        Tensor({}, dtype),
        Tensor({}, dtype),
        Tensor({}, ge::DT_FLOAT),
    };
    std::vector<Desc> outputs = {
        Tensor(SoftmaxShape(layout, b, n1, s1), ge::DT_FLOAT),
        Tensor(SoftmaxShape(layout, b, n1, s1), ge::DT_FLOAT),
        Tensor({}, dtype),
        Tensor(DataShape(layout, b, n1, s1, dv), dtype),
    };
    const float scale = 1.0F / std::sqrt(static_cast<float>(d));
    const int64_t allTokens = std::numeric_limits<int32_t>::max();
    std::vector<Attr> attrs = {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(scale)},
        {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(hasDrop ? 0.9F : 1.0F)},
        {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(allTokens)},
        {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(allTokens)},
        {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n1)},
        {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(LayoutName(layout))},
        {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(
             layout == 3 ? "same_as_input" : "")},
    };

    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        24, 48, 196608, 524288, 131072, 201326592,
        platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara context(
        "FlashAttentionScore", inputs, outputs, attrs, &compileInfo,
        "Ascend910B", 64, 262144, 8192);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    PrintResult(info);
}

}  // namespace
