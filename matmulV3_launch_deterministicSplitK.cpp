#define DTYPE_X1 half
#define DTYPE_X2 half
#define DTYPE_Y float
#define DTYPE_BIAS float
#ifndef ORIG_DTYPE_X1
#define ORIG_DTYPE_X1 DT_FLOAT16
#endif
#ifndef ORIG_DTYPE_BIAS
#define ORIG_DTYPE_BIAS DT_FLOAT
#endif

#include "kernel_operator.h"
#include "op_kernel/mat_mul_v3_common.h"
#include "op_kernel/mat_mul_base_block.h"
#include "op_kernel/mat_mul_unaligned_deterministic_splitk_kernel.h"
#include "op_kernel/mat_mul_v3_cycles.h"

using namespace AscendC;
using namespace matmul;

__aicore__ inline void CopyTilingFromGM(MatmulTilingData &dst, GM_ADDR tilingGM)
{
    auto src = reinterpret_cast<__gm__ uint32_t *>(tilingGM);
    auto dp  = reinterpret_cast<uint32_t *>(&dst);
    for (uint32_t i = 0; i < sizeof(MatmulTilingData) / sizeof(uint32_t); ++i) dp[i] = src[i];
}

extern "C" __global__ __aicore__ void matmul_v3_deterministic_splitK(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                                                     GM_ADDR offsetWGM, GM_ADDR cGM,
                                                                     GM_ADDR workspaceGM, GM_ADDR tilingGM,
                                                                     GM_ADDR cycleGM)
{
    MatmulTilingData tilingData;
    CopyTilingFromGM(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    using aType    = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, false>;
    using bType    = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
    using cType    = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>;
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;

    pipe_barrier(PIPE_ALL);
    uint64_t startCycle = AscendC::GetSystemCycle();
    MatMulUnAlignedKernelDeterministicSplitK<aType, bType, cType, biasType, FIXPIPE_OPT_SELECT::BASE>(
        aGM, bGM, cGM, biasGM, tilingData, user);
    pipe_barrier(PIPE_ALL);
    MatmulV3Cycles::Write(cycleGM, AscendC::GetSystemCycle() - startCycle);
}