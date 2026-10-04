#include <acl/acl.h>
#ifndef ATTENTION_GRAD_ONLY
#include "aclnn_flash_attention_score.h"
#endif
#ifndef ATTENTION_FORWARD_ONLY
#include "aclnn_flash_attention_score_grad.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

void *LoadCustomTilingLibrary()
{
    const char *root = std::getenv("ATTENTION_CUSTOM_OPP_ROOT");
    if (root == nullptr || *root == '\0') {
        std::fprintf(stderr, "[ERROR] ATTENTION_CUSTOM_OPP_ROOT is not set\n");
        return nullptr;
    }

    const std::vector<std::string> candidates = {
        std::string(root) + "/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so",
        std::string(root) + "/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so",
        std::string(root) + "/op_impl/ai_core/tbe/op_tiling/liboptiling.so",
    };
    for (const std::string &path : candidates) {
        if (access(path.c_str(), R_OK) != 0) {
            continue;
        }
        dlerror();
        void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle != nullptr) {
            std::fprintf(stdout, "[INFO] loaded custom Host tiling library: %s\n", path.c_str());
            return handle;
        }
        std::fprintf(stderr, "[ERROR] dlopen(%s) failed: %s\n", path.c_str(), dlerror());
        return nullptr;
    }
    std::fprintf(stderr, "[ERROR] custom Host tiling library is missing under %s\n", root);
    return nullptr;
}

int64_t EnvI64(const char *name, int64_t fallback)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const long long value = std::strtoll(raw, &end, 10);
    if (end == raw || *end != '\0') {
        std::fprintf(stderr, "[ERROR] invalid integer %s=%s\n", name, raw);
        std::exit(2);
    }
    return static_cast<int64_t>(value);
}

int64_t Numel(const std::vector<int64_t> &shape)
{
    int64_t result = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0 || result > std::numeric_limits<int64_t>::max() / dim) {
            std::fprintf(stderr, "[ERROR] invalid or overflowing tensor shape\n");
            std::exit(2);
        }
        result *= dim;
    }
    return result;
}

struct Tensor {
    aclTensor *tensor = nullptr;
    void *device = nullptr;
    size_t bytes = 0;
};

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000U;
    const uint32_t mantissa = bits & 0x7fffffU;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        const uint32_t normalized = (mantissa | 0x800000U) >> (1 - exponent);
        return static_cast<uint16_t>(sign | ((normalized + 0x1000U) >> 13));
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U | (mantissa != 0 ? 0x0200U : 0));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                                 ((mantissa + 0x1000U) >> 13));
}

float HalfToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    uint32_t exponent = (value >> 10) & 0x1fU;
    uint32_t mantissa = value & 0x03ffU;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffU;
            bits = sign | ((127U - 14U - static_cast<uint32_t>(shift)) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000U | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127U - 15U) << 23) | (mantissa << 13);
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToBFloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t roundingBias = 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>((bits + roundingBias) >> 16);
}

float BFloat16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int MakeTensor(const std::vector<int64_t> &shape, aclDataType dtype, size_t elementBytes, Tensor &out)
{
    out.bytes = static_cast<size_t>(Numel(shape)) * elementBytes;
    aclError ret = aclrtMalloc(&out.device, out.bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "[ERROR] aclrtMalloc(%zu) failed: %d\n", out.bytes, ret);
        return ret;
    }
    ret = aclrtMemset(out.device, out.bytes, 0, out.bytes);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "[ERROR] aclrtMemset failed: %d\n", ret);
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    out.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                 shape.data(), shape.size(), out.device);
    if (out.tensor == nullptr) {
        std::fprintf(stderr, "[ERROR] aclCreateTensor failed\n");
        return 1;
    }
    return ACL_SUCCESS;
}

int FillFloat(Tensor &tensor, float value)
{
    std::vector<float> host(tensor.bytes / sizeof(float), value);
    const aclError ret = aclrtMemcpy(tensor.device, tensor.bytes, host.data(), tensor.bytes,
                                     ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "[ERROR] aclrtMemcpy input failed: %d\n", ret);
    }
    return ret;
}

int FillData(Tensor &tensor, aclDataType dtype, float base)
{
    const size_t elements = dtype == ACL_FLOAT ? tensor.bytes / sizeof(float)
                                               : tensor.bytes / sizeof(uint16_t);
    aclError ret = ACL_SUCCESS;
    if (dtype == ACL_FLOAT) {
        std::vector<float> host(elements);
        for (size_t i = 0; i < elements; ++i) {
            host[i] = base + static_cast<float>(static_cast<int>(i % 17) - 8) * 0.001F;
        }
        ret = aclrtMemcpy(tensor.device, tensor.bytes, host.data(), tensor.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    } else {
        std::vector<uint16_t> host(elements);
        for (size_t i = 0; i < elements; ++i) {
            const float value = base + static_cast<float>(static_cast<int>(i % 17) - 8) * 0.001F;
            host[i] = dtype == ACL_BF16 ? FloatToBFloat16(value) : FloatToHalf(value);
        }
        ret = aclrtMemcpy(tensor.device, tensor.bytes, host.data(), tensor.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "[ERROR] aclrtMemcpy input failed: %d\n", ret);
    }
    return ret;
}

int FillDropMask(Tensor &tensor)
{
    std::vector<uint8_t> host(tensor.bytes);
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = (i % 2 == 0) ? 0xAAU : 0xD5U;
    }
    const aclError ret = aclrtMemcpy(tensor.device, tensor.bytes, host.data(), host.size(),
                                     ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "[ERROR] drop-mask copy failed: %d\n", ret);
    }
    return ret;
}

std::vector<int64_t> DataShape(int64_t layout, int64_t batch, int64_t heads, int64_t seq, int64_t dim)
{
    switch (layout) {
        case 0: return {batch, heads, seq, dim};       // BNSD
        case 1: return {seq, batch, heads * dim};      // SBH
        case 2: return {batch, seq, heads, dim};       // BSND
        case 3: return {batch * seq, heads, dim};      // TND (uniform sequences)
        default:
            std::fprintf(stderr, "[ERROR] unsupported ATTENTION_LAYOUT=%ld\n", layout);
            std::exit(2);
    }
}

const char *LayoutName(int64_t layout)
{
    switch (layout) {
        case 0: return "BNSD";
        case 1: return "SBH";
        case 2: return "BSND";
        case 3: return "TND";
        default: return "";
    }
}

std::vector<int64_t> SoftmaxShape(int64_t layout, int64_t batch, int64_t heads, int64_t seq)
{
    return layout == 3 ? std::vector<int64_t>{batch * seq, heads, 8}
                       : std::vector<int64_t>{batch, heads, seq, 8};
}

int DumpOutputs(const std::vector<const Tensor *> &outputs, aclDataType dtype)
{
    mkdir("output", 0755);
    std::ofstream file("output/output.bin", std::ios::binary | std::ios::trunc);
    for (const Tensor *output : outputs) {
        std::vector<uint8_t> host(output->bytes);
        const aclError ret = aclrtMemcpy(host.data(), host.size(), output->device, output->bytes,
                                         ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "[ERROR] output copy failed: %d\n", ret);
            return ret;
        }
        if (dtype == ACL_FLOAT) {
            file.write(reinterpret_cast<const char *>(host.data()), static_cast<std::streamsize>(host.size()));
        } else {
            const auto *halves = reinterpret_cast<const uint16_t *>(host.data());
            std::vector<float> converted(host.size() / sizeof(uint16_t));
            for (size_t i = 0; i < converted.size(); ++i) {
                converted[i] = dtype == ACL_BF16 ? BFloat16ToFloat(halves[i]) : HalfToFloat(halves[i]);
            }
            file.write(reinterpret_cast<const char *>(converted.data()),
                       static_cast<std::streamsize>(converted.size() * sizeof(float)));
        }
    }
    return file.good() ? 0 : 1;
}

void Destroy(Tensor &value)
{
    if (value.tensor != nullptr) {
        aclDestroyTensor(value.tensor);
    }
    if (value.device != nullptr) {
        aclrtFree(value.device);
    }
}

int CheckFeatures(const char *prefix)
{
    const char *unsupported[] = {"HAS_MASK", "HAS_PSE", "HAS_ROPE", "HAS_SINK", "HAS_START_IDX"};
    for (const char *feature : unsupported) {
        const std::string name = std::string(prefix) + "_" + feature;
        if (EnvI64(name.c_str(), 0) != 0) {
            std::fprintf(stderr, "[ERROR] the bundled benchmark does not yet construct %s inputs\n", name.c_str());
            return 2;
        }
    }
    return 0;
}

#ifndef ATTENTION_GRAD_ONLY
int RunForward(aclrtStream stream, int64_t b, int64_t n1, int64_t n2, int64_t s1, int64_t s2,
               int64_t d, int64_t dv, int64_t layout, aclDataType dtype, size_t typeBytes)
{
    if (CheckFeatures("FA") != 0) {
        return 2;
    }
    Tensor q, k, v, dropMask, softmaxMax, softmaxSum, attention;
    if (MakeTensor(DataShape(layout, b, n1, s1, d), dtype, typeBytes, q) != ACL_SUCCESS ||
        MakeTensor(DataShape(layout, b, n2, s2, d), dtype, typeBytes, k) != ACL_SUCCESS ||
        MakeTensor(DataShape(layout, b, n2, s2, dv), dtype, typeBytes, v) != ACL_SUCCESS ||
        MakeTensor(SoftmaxShape(layout, b, n1, s1), ACL_FLOAT, sizeof(float), softmaxMax) != ACL_SUCCESS ||
        MakeTensor(SoftmaxShape(layout, b, n1, s1), ACL_FLOAT, sizeof(float), softmaxSum) != ACL_SUCCESS ||
        MakeTensor(DataShape(layout, b, n1, s1, dv), dtype, typeBytes, attention) != ACL_SUCCESS) {
        return 1;
    }
    if (FillData(q, dtype, 0.01F) != ACL_SUCCESS || FillData(k, dtype, 0.02F) != ACL_SUCCESS ||
        FillData(v, dtype, 0.03F) != ACL_SUCCESS) {
        return 1;
    }

    const bool hasDrop = EnvI64("FA_HAS_DROP", 0) != 0;
    if (hasDrop) {
        const int64_t dropElements = (b * n1 * s1 * s2 + 7) / 8;
        if (MakeTensor({dropElements}, ACL_UINT8, sizeof(uint8_t), dropMask) != ACL_SUCCESS) {
            return 1;
        }
        if (FillDropMask(dropMask) != ACL_SUCCESS) {
            return 1;
        }
    }
    const double keepProb = hasDrop ? 0.9 : 1.0;

    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    const int64_t allTokens = std::numeric_limits<int32_t>::max();
    char layoutName[5] = {};
    std::strncpy(layoutName, LayoutName(layout), sizeof(layoutName) - 1);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclIntArray *actualQ = nullptr;
    aclIntArray *actualKv = nullptr;
    std::vector<int64_t> actualQData;
    std::vector<int64_t> actualKvData;
    aclnnStatus ret;

    if (layout == 3) {
        for (int64_t i = 1; i <= b; ++i) {
            actualQData.push_back(i * s1);
            actualKvData.push_back(i * s2);
        }
        actualQ = aclCreateIntArray(actualQData.data(), actualQData.size());
        actualKv = aclCreateIntArray(actualKvData.data(), actualKvData.size());
        char softmaxLayout[] = "same_as_input";
        ret = aclnnFlashAttentionVarLenScoreV4GetWorkspaceSize(
            q.tensor, k.tensor, v.tensor, nullptr, dropMask.tensor, nullptr, nullptr, nullptr, actualQ, actualKv,
            scale, keepProb, allTokens, allTokens, n1, layoutName, 0, 0, softmaxLayout,
            softmaxMax.tensor, softmaxSum.tensor, nullptr, attention.tensor, &workspaceSize, &executor);
    } else {
        ret = aclnnFlashAttentionScoreV2GetWorkspaceSize(
            q.tensor, k.tensor, v.tensor, nullptr, dropMask.tensor, nullptr, nullptr, nullptr, nullptr, nullptr,
            scale, keepProb, allTokens, allTokens, n1, layoutName, 0, 0, 1,
            softmaxMax.tensor, softmaxSum.tensor, nullptr, attention.tensor, &workspaceSize, &executor);
    }
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "[ERROR] FA GetWorkspaceSize failed: %d, %s\n", ret, aclGetRecentErrMsg());
        return ret;
    }

    void *workspace = nullptr;
    if (EnvI64("ATTENTION_DISCOVER_ONLY", 0) == 0) {
        if (workspaceSize != 0 && aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            return 1;
        }
        ret = layout == 3 ? aclnnFlashAttentionVarLenScoreV4(workspace, workspaceSize, executor, stream)
                          : aclnnFlashAttentionScoreV2(workspace, workspaceSize, executor, stream);
        if (ret == ACL_SUCCESS) {
            ret = aclrtSynchronizeStream(stream);
        }
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "[ERROR] FA execution failed: %d, %s\n", ret, aclGetRecentErrMsg());
        } else {
            ret = DumpOutputs({&attention}, dtype);
        }
    }

    if (workspace != nullptr) aclrtFree(workspace);
    if (actualQ != nullptr) aclDestroyIntArray(actualQ);
    if (actualKv != nullptr) aclDestroyIntArray(actualKv);
    Destroy(q); Destroy(k); Destroy(v); Destroy(dropMask); Destroy(softmaxMax); Destroy(softmaxSum); Destroy(attention);
    return ret;
}
#endif

#ifndef ATTENTION_FORWARD_ONLY
int RunBackward(aclrtStream stream, int64_t b, int64_t n1, int64_t n2, int64_t s1, int64_t s2,
                int64_t d, int64_t dv, int64_t layout, aclDataType dtype, size_t typeBytes)
{
    if (CheckFeatures("FAG") != 0) {
        return 2;
    }
    if (EnvI64("FAG_HAS_DROP", 0) != 0) {
        std::fprintf(stderr, "[ERROR] the bundled backward benchmark does not construct FAG_HAS_DROP inputs\n");
        return 2;
    }
    Tensor q, k, v, dy, softmaxMax, softmaxSum, attention, dq, dk, dvOut;
    if (MakeTensor(DataShape(layout, b, n1, s1, d), dtype, typeBytes, q) != ACL_SUCCESS ||
        MakeTensor(DataShape(layout, b, n2, s2, d), dtype, typeBytes, k) != ACL_SUCCESS ||
        MakeTensor(DataShape(layout, b, n2, s2, dv), dtype, typeBytes, v) != ACL_SUCCESS ||
        MakeTensor(DataShape(layout, b, n1, s1, dv), dtype, typeBytes, dy) != ACL_SUCCESS ||
        MakeTensor(SoftmaxShape(layout, b, n1, s1), ACL_FLOAT, sizeof(float), softmaxMax) != ACL_SUCCESS ||
        MakeTensor(SoftmaxShape(layout, b, n1, s1), ACL_FLOAT, sizeof(float), softmaxSum) != ACL_SUCCESS ||
        MakeTensor(DataShape(layout, b, n1, s1, dv), dtype, typeBytes, attention) != ACL_SUCCESS ||
        MakeTensor(DataShape(layout, b, n1, s1, d), dtype, typeBytes, dq) != ACL_SUCCESS ||
        MakeTensor(DataShape(layout, b, n2, s2, d), dtype, typeBytes, dk) != ACL_SUCCESS ||
        MakeTensor(DataShape(layout, b, n2, s2, dv), dtype, typeBytes, dvOut) != ACL_SUCCESS) {
        return 1;
    }
    if (FillData(q, dtype, 0.01F) != ACL_SUCCESS || FillData(k, dtype, 0.02F) != ACL_SUCCESS ||
        FillData(v, dtype, 0.03F) != ACL_SUCCESS || FillData(dy, dtype, 0.04F) != ACL_SUCCESS ||
        FillData(attention, dtype, 0.05F) != ACL_SUCCESS || FillFloat(softmaxMax, 3.0F) != ACL_SUCCESS ||
        FillFloat(softmaxSum, 3.0F) != ACL_SUCCESS) {
        return 1;
    }

    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    const int64_t allTokens = std::numeric_limits<int32_t>::max();
    char layoutName[5] = {};
    std::strncpy(layoutName, LayoutName(layout), sizeof(layoutName) - 1);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclIntArray *actualQ = nullptr;
    aclIntArray *actualKv = nullptr;
    std::vector<int64_t> actualQData;
    std::vector<int64_t> actualKvData;
    aclnnStatus ret;

    if (layout == 3) {
        for (int64_t i = 1; i <= b; ++i) {
            actualQData.push_back(i * s1);
            actualKvData.push_back(i * s2);
        }
        actualQ = aclCreateIntArray(actualQData.data(), actualQData.size());
        actualKv = aclCreateIntArray(actualKvData.data(), actualKvData.size());
        char softmaxLayout[] = "same_as_input";
        ret = aclnnFlashAttentionUnpaddingScoreGradV4GetWorkspaceSize(
            q.tensor, k.tensor, v.tensor, dy.tensor, nullptr, nullptr, nullptr, nullptr,
            softmaxMax.tensor, softmaxSum.tensor, nullptr, attention.tensor, nullptr, actualQ, actualKv,
            scale, 1.0, allTokens, allTokens, n1, layoutName, 0, 0,
            dq.tensor, dk.tensor, dvOut.tensor, nullptr, softmaxLayout, &workspaceSize, &executor);
    } else {
        ret = aclnnFlashAttentionScoreGradV3GetWorkspaceSize(
            q.tensor, k.tensor, v.tensor, dy.tensor, nullptr, nullptr, nullptr, nullptr,
            softmaxMax.tensor, softmaxSum.tensor, nullptr, attention.tensor, nullptr, nullptr,
            nullptr, nullptr, scale, 1.0, allTokens, allTokens, n1, layoutName, 0, 0, 1,
            dq.tensor, dk.tensor, dvOut.tensor, nullptr, nullptr, &workspaceSize, &executor);
    }
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "[ERROR] FAG GetWorkspaceSize failed: %d, %s\n", ret, aclGetRecentErrMsg());
        return ret;
    }

    void *workspace = nullptr;
    if (EnvI64("ATTENTION_DISCOVER_ONLY", 0) == 0) {
        if (workspaceSize != 0 && aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            return 1;
        }
        ret = layout == 3 ? aclnnFlashAttentionUnpaddingScoreGradV4(workspace, workspaceSize, executor, stream)
                          : aclnnFlashAttentionScoreGradV3(workspace, workspaceSize, executor, stream);
        if (ret == ACL_SUCCESS) {
            ret = aclrtSynchronizeStream(stream);
        }
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "[ERROR] FAG execution failed: %d, %s\n", ret, aclGetRecentErrMsg());
        } else {
            ret = DumpOutputs({&dq, &dk, &dvOut}, dtype);
        }
    }

    if (workspace != nullptr) aclrtFree(workspace);
    if (actualQ != nullptr) aclDestroyIntArray(actualQ);
    if (actualKv != nullptr) aclDestroyIntArray(actualKv);
    Destroy(q); Destroy(k); Destroy(v); Destroy(dy); Destroy(softmaxMax); Destroy(softmaxSum);
    Destroy(attention); Destroy(dq); Destroy(dk); Destroy(dvOut);
    return ret;
}
#endif

}  // namespace

int main()
{
    // Load the cached custom Host tiling implementation before ACLNN resolves
    // FlashAttentionScore.  Candidate values are consumed at runtime, so a
    // new shape or PSO proposal must not trigger another operator build.
    void *tilingLibrary = LoadCustomTilingLibrary();
    if (tilingLibrary == nullptr) {
        return 2;
    }

    const bool grad = EnvI64("ATTENTION_GRAD", 0) != 0;
    const char *prefix = grad ? "FAG" : "FA";
    const auto parameter = [prefix](const char *suffix, int64_t fallback) {
        return EnvI64((std::string(prefix) + "_" + suffix).c_str(), fallback);
    };
    const int64_t b = parameter("B", 1);
    const int64_t n1 = parameter("N1", 8);
    const int64_t n2 = parameter("N2", 1);
    const int64_t s1 = parameter("S1", 128);
    const int64_t s2 = parameter("S2", 1536);
    const int64_t d = parameter("D", 128);
    const int64_t dv = parameter("DV", d);
    const int64_t layout = parameter("LAYOUT", 0);
    const int64_t dtypeBytes = parameter("DTYPE_BYTES", 4);
    const int64_t dtypeKind = parameter("DTYPE_KIND", 0);
    if (b <= 0 || n1 <= 0 || n2 <= 0 || n1 % n2 != 0 || s1 <= 0 || s2 <= 0 || d <= 0 || dv <= 0 ||
        (dtypeBytes != 2 && dtypeBytes != 4) || (dtypeKind != 0 && dtypeKind != 2) ||
        (dtypeKind == 2 && dtypeBytes != 2)) {
        std::fprintf(stderr, "[ERROR] invalid attention shape or dtype\n");
        return 2;
    }
    const aclDataType dtype = dtypeKind == 2 ? ACL_BF16 : (dtypeBytes == 4 ? ACL_FLOAT : ACL_FLOAT16);

    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) return ret;
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) return ret;
    ret = aclrtCtxSetSysParamOpt(ACL_OPT_DETERMINISTIC, parameter("DETERMINISTIC", 0));
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "[ERROR] failed to set deterministic mode: %d\n", ret);
        return ret;
    }
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) return ret;

    int result = 0;
#ifdef ATTENTION_GRAD_ONLY
    if (!grad) {
        std::fprintf(stderr, "[ERROR] this launcher was built for FlashAttentionScoreGrad\n");
        result = 2;
    } else {
        result = RunBackward(stream, b, n1, n2, s1, s2, d, dv, layout, dtype, dtypeBytes);
    }
#elif defined(ATTENTION_FORWARD_ONLY)
    if (grad) {
        std::fprintf(stderr, "[ERROR] this launcher was built for FlashAttentionScore\n");
        result = 2;
    } else {
        result = RunForward(stream, b, n1, n2, s1, s2, d, dv, layout, dtype, dtypeBytes);
    }
#else
    result = grad ? RunBackward(stream, b, n1, n2, s1, s2, d, dv, layout, dtype, dtypeBytes)
                  : RunForward(stream, b, n1, n2, s1, s2, d, dv, layout, dtype, dtypeBytes);
#endif
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    dlclose(tilingLibrary);
    return result;
}
