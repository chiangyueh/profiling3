// NEW BEGIN
#include <acl/acl.h>
#include "aclnn_flash_attention_score.h"

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

int64_t EnvI64(const char *name, int64_t fallback)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const long long value = std::strtoll(raw, &end, 10);
    if (end == raw || *end != '\0') {
        std::fprintf(stderr, "invalid integer %s=%s\n", name, raw);
        std::exit(2);
    }
    return static_cast<int64_t>(value);
}

void *LoadHostLibrary()
{
    const char *root = std::getenv("FA_CUSTOM_OPP_ROOT");
    if (root == nullptr || *root == '\0') {
        return nullptr;
    }
    const std::vector<std::string> paths = {
        std::string(root) + "/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so",
        std::string(root) + "/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so",
        std::string(root) + "/op_impl/ai_core/tbe/op_tiling/liboptiling.so",
    };
    for (const auto &path : paths) {
        if (access(path.c_str(), R_OK) == 0) {
            return dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        }
    }
    return nullptr;
}

int64_t Numel(const std::vector<int64_t> &shape)
{
    int64_t value = 1;
    for (int64_t dim : shape) {
        if (dim <= 0 || value > std::numeric_limits<int64_t>::max() / dim) {
            std::exit(2);
        }
        value *= dim;
    }
    return value;
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
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffU) - 112;
    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
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
            bits = sign | ((113U - static_cast<uint32_t>(shift)) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000U | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToBFloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
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
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    out.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                 shape.data(), shape.size(), out.device);
    return out.tensor == nullptr ? 1 : ACL_SUCCESS;
}

int Fill(Tensor &tensor, aclDataType dtype, float base)
{
    const size_t elements = dtype == ACL_FLOAT ? tensor.bytes / sizeof(float)
                                               : tensor.bytes / sizeof(uint16_t);
    if (dtype == ACL_FLOAT) {
        std::vector<float> host(elements);
        for (size_t i = 0; i < elements; ++i) {
            host[i] = base + static_cast<float>(static_cast<int>(i % 17) - 8) * 0.001F;
        }
        return aclrtMemcpy(tensor.device, tensor.bytes, host.data(), tensor.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    std::vector<uint16_t> host(elements);
    for (size_t i = 0; i < elements; ++i) {
        const float value = base + static_cast<float>(static_cast<int>(i % 17) - 8) * 0.001F;
        host[i] = dtype == ACL_BF16 ? FloatToBFloat16(value) : FloatToHalf(value);
    }
    return aclrtMemcpy(tensor.device, tensor.bytes, host.data(), tensor.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
}

std::vector<int64_t> Shape(int64_t layout, int64_t b, int64_t n, int64_t s, int64_t d)
{
    if (layout == 0) {
        return {b, n, s, d};
    }
    if (layout == 1) {
        return {s, b, n * d};
    }
    if (layout == 2) {
        return {b, s, n, d};
    }
    std::exit(2);
}

const char *Layout(int64_t layout)
{
    if (layout == 0) {
        return "BNSD";
    }
    if (layout == 1) {
        return "SBH";
    }
    if (layout == 2) {
        return "BSND";
    }
    return "";
}

void Destroy(Tensor &tensor)
{
    if (tensor.tensor != nullptr) {
        aclDestroyTensor(tensor.tensor);
    }
    if (tensor.device != nullptr) {
        aclrtFree(tensor.device);
    }
}

int Dump(Tensor &tensor, aclDataType dtype)
{
    mkdir("output", 0755);
    std::vector<uint8_t> host(tensor.bytes);
    aclError ret = aclrtMemcpy(host.data(), host.size(), tensor.device, tensor.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    std::ofstream file("output/output.bin", std::ios::binary | std::ios::trunc);
    if (dtype == ACL_FLOAT) {
        file.write(reinterpret_cast<const char *>(host.data()), static_cast<std::streamsize>(host.size()));
    } else {
        const auto *input = reinterpret_cast<const uint16_t *>(host.data());
        std::vector<float> output(host.size() / sizeof(uint16_t));
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] = dtype == ACL_BF16 ? BFloat16ToFloat(input[i]) : HalfToFloat(input[i]);
        }
        file.write(reinterpret_cast<const char *>(output.data()),
                   static_cast<std::streamsize>(output.size() * sizeof(float)));
    }
    return file.good() ? 0 : 1;
}

int Run(aclrtStream stream)
{
    const int64_t b = EnvI64("FA_B", 1);
    const int64_t n1 = EnvI64("FA_N1", 64);
    const int64_t n2 = EnvI64("FA_N2", 64);
    const int64_t s1 = EnvI64("FA_S1", 4096);
    const int64_t s2 = EnvI64("FA_S2", 4096);
    const int64_t d = EnvI64("FA_D", 192);
    const int64_t dv = EnvI64("FA_DV", 128);
    const int64_t layout = EnvI64("FA_LAYOUT", 1);
    const int64_t dtypeKind = EnvI64("FA_DTYPE_KIND", 2);
    const aclDataType dtype = dtypeKind == 2 ? ACL_BF16 : (dtypeKind == 1 ? ACL_FLOAT16 : ACL_FLOAT);
    const size_t typeBytes = dtype == ACL_FLOAT ? sizeof(float) : sizeof(uint16_t);
    if (b <= 0 || n1 <= 0 || n2 <= 0 || n1 % n2 != 0 || s1 <= 0 || s2 <= 0 || d <= 0 || dv <= 0 ||
        layout < 0 || layout > 2 || dtypeKind < 0 || dtypeKind > 2) {
        return 2;
    }

    Tensor q, k, v, softmaxMax, softmaxSum, attention;
    if (MakeTensor(Shape(layout, b, n1, s1, d), dtype, typeBytes, q) != ACL_SUCCESS ||
        MakeTensor(Shape(layout, b, n2, s2, d), dtype, typeBytes, k) != ACL_SUCCESS ||
        MakeTensor(Shape(layout, b, n2, s2, dv), dtype, typeBytes, v) != ACL_SUCCESS ||
        MakeTensor({b, n1, s1, 8}, ACL_FLOAT, sizeof(float), softmaxMax) != ACL_SUCCESS ||
        MakeTensor({b, n1, s1, 8}, ACL_FLOAT, sizeof(float), softmaxSum) != ACL_SUCCESS ||
        MakeTensor(Shape(layout, b, n1, s1, dv), dtype, typeBytes, attention) != ACL_SUCCESS) {
        return 1;
    }
    if (Fill(q, dtype, 0.01F) != ACL_SUCCESS || Fill(k, dtype, 0.02F) != ACL_SUCCESS ||
        Fill(v, dtype, 0.03F) != ACL_SUCCESS) {
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    const int64_t tokens = std::numeric_limits<int32_t>::max();
    char layoutName[5] = {};
    std::strncpy(layoutName, Layout(layout), sizeof(layoutName) - 1);
    aclnnStatus ret = aclnnFlashAttentionScoreV2GetWorkspaceSize(
        q.tensor, k.tensor, v.tensor, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        scale, 1.0, tokens, tokens, n1, layoutName, 0, 0, 1,
        softmaxMax.tensor, softmaxSum.tensor, nullptr, attention.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "GetWorkspaceSize failed: %d, %s\n", ret, aclGetRecentErrMsg());
        return ret;
    }

    void *workspace = nullptr;
    if (EnvI64("FA_DISCOVER_ONLY", 0) == 0) {
        if (workspaceSize != 0 && aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            return 1;
        }
        ret = aclnnFlashAttentionScoreV2(workspace, workspaceSize, executor, stream);
        if (ret == ACL_SUCCESS) {
            ret = aclrtSynchronizeStream(stream);
        }
        if (ret == ACL_SUCCESS) {
            ret = Dump(attention, dtype);
        } else {
            std::fprintf(stderr, "execution failed: %d, %s\n", ret, aclGetRecentErrMsg());
        }
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    Destroy(q);
    Destroy(k);
    Destroy(v);
    Destroy(softmaxMax);
    Destroy(softmaxSum);
    Destroy(attention);
    return ret;
}

}

int main()
{
    void *hostLibrary = LoadHostLibrary();
    if (hostLibrary == nullptr) {
        return 2;
    }
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    const int result = Run(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    dlclose(hostLibrary);
    return result;
}
// NEW END
