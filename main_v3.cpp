#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mat_mul_v3_tiling_data.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#else
#include "tikicpulib.h"
#endif
#include "matmul_v3_kernels.h"
static int32_t EnvI(const char *name, int32_t def)
{
    const char *v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}
static uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

#ifndef ASCENDC_CPU_DEBUG
static void PrintPlatformInfo()
{
    const char *soc = aclrtGetSocName();
    printf("socVersion       : %s\n", soc ? soc : "(null)");

    auto p = platform_ascendc::PlatformAscendCManager::GetInstance(soc);
    if (p == nullptr) {
        printf("[ERROR] PlatformAscendCManager::GetInstance failed\n");
        return;
    }

    printf("AIC (cube) cores : %u\n", p->GetCoreNumAic());
    printf("AIV (vector)     : %u\n", p->GetCoreNumAiv());

    uint64_t l0a = 0, l0b = 0, l0c = 0, l1 = 0, ub = 0;
    p->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
    p->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
    p->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    p->GetCoreMemSize(platform_ascendc::CoreMemType::L1,   l1);
    p->GetCoreMemSize(platform_ascendc::CoreMemType::UB,   ub);

    printf("L0A / L0B / L0C  : %lu / %lu / %lu B\n", l0a, l0b, l0c);
    printf("L1 / UB          : %lu / %lu B\n", l1, ub);
}
#endif

int32_t main(int32_t argc, char *argv[])
{
    // 0 = base, 1 = single-core split-K, 2 = deterministic split-K
    const int32_t kernelType = EnvI("MM_KERNEL_TYPE", 0);
    const int32_t M = EnvI("MM_M", 4096);
    const int32_t N = EnvI("MM_N", 7168);
    const int32_t K = EnvI("MM_K", 32768);

    const int32_t usedCoreNum = EnvI("MM_CORE_NUM", 24);
    const int32_t singleCoreM = EnvI("MM_SINGLE_M", 512);
    const int32_t singleCoreN = EnvI("MM_SINGLE_N", 2432);
    const int32_t singleCoreK = EnvI("MM_SINGLE_K", 384);
    const int32_t baseM = EnvI("MM_BASE_M", 128);
    const int32_t baseN = EnvI("MM_BASE_N", 128);
    const int32_t baseK = EnvI("MM_BASE_K", 128);
    const int32_t stepM = EnvI("MM_STEP_M", 3);
    const int32_t stepN = EnvI("MM_STEP_N", 1);
    const int32_t stepKa = EnvI("MM_STEP_Ka", 3);
    const int32_t stepKb = EnvI("MM_STEP_Kb", 3);
    const int32_t dbL0A = EnvI("MM_DB_L0A", 2);
    const int32_t dbL0B = EnvI("MM_DB_L0B", 2);
    const int32_t dbL0C = EnvI("MM_DB_L0C", 2);
    const int32_t iterateOrder = EnvI("MM_ITERATE_ORDER", 0);
    const int32_t mTileBlock = EnvI("MM_M_TILE_BLOCK", 4);
    const int32_t nTileBlock = EnvI("MM_N_TILE_BLOCK", 4);
    const int32_t calOrder = EnvI("MM_CAL_ORDER", 0);

    const int32_t nd2nzA = EnvI("MM_ND2NZ_A", 0);
    const int32_t nd2nzB = EnvI("MM_ND2NZ_B", 0);
    const int32_t baseAN = EnvI("MM_ND2NZ_BASE_AN", 0);
    const int32_t baseAD = EnvI("MM_ND2NZ_BASE_AD", 0);
    const int32_t baseBN = EnvI("MM_ND2NZ_BASE_BN", 0);
    const int32_t baseBD = EnvI("MM_ND2NZ_BASE_BD", 0);

    // db для L1 не всегда 2: SetBasicBlockOfMK33 ставит depthA1 = stepM*stepKa*1,
    // depthB1 = stepN*stepKb*2 (у NK33 наоборот). Поэтому переопределяемо.
    const int32_t depthA1 = EnvI("MM_DEPTH_A1", stepM * stepKa * 2);
    const int32_t depthB1 = EnvI("MM_DEPTH_B1", stepN * stepKb * 2);
    const int32_t mTileCntL2 = CeilDiv(CeilDiv(M, baseM), mTileBlock);
    const int32_t nTileCntL2 = CeilDiv(CeilDiv(N, baseN), nTileBlock);

    const int32_t isBias = EnvI("MM_IS_BIAS", 0);
    const int32_t transLength = EnvI("MM_TRANS_LENGTH", 0);
    const int32_t transA = EnvI("MM_TRANS_A", 0);
    const int32_t transB = EnvI("MM_TRANS_B", 0);
    const int32_t isNzA = EnvI("MM_IS_NZ_A", 0);
    const int32_t isNzB = EnvI("MM_IS_NZ_B", 0);
    const int32_t isHf32 = EnvI("MM_IS_HF32", 0);

    // Замер системным счётчиком вместо профилировщика: включается только
    // флагом --cycles-only у run.sh, который выставляет MM_CYCLES=1.
    const int32_t cyclesMode = EnvI("MM_CYCLES", 0);

    size_t aSize = static_cast<size_t>(M) * K * sizeof(uint16_t);
    size_t bSize = static_cast<size_t>(K) * N * sizeof(uint16_t);
    size_t cSize = static_cast<size_t>(M) * N * sizeof(float);
    size_t tilingSize = sizeof(MatmulTilingData);

    const size_t RPC = 20ull * 1024 * 1024;
    size_t detWs = (kernelType == 2)
        ? (size_t)usedCoreNum * singleCoreM * singleCoreN * 2 * sizeof(float)
        : 0;
    // под сконвертированные ND->NZ операнды (transA = transB = 0)
    const size_t c0 = 32 / sizeof(uint16_t);
    size_t nd2nzWs = 0;
    if (nd2nzA) {
        nd2nzWs += (size_t)CeilDiv(M, 16) * 16 * CeilDiv(K, c0) * c0 * sizeof(uint16_t);
    }
    if (nd2nzB) {
        nd2nzWs += (size_t)CeilDiv(N, c0) * c0 * CeilDiv(K, 16) * 16 * sizeof(uint16_t);
    }
    size_t sysWorkspaceSize = RPC + detWs + nd2nzWs;
    uint8_t *tilingHost = (uint8_t *)malloc(tilingSize);
    memset(tilingHost, 0, tilingSize);
    MatmulTilingData *t = reinterpret_cast<MatmulTilingData *>(tilingHost);
    // matmulTiling
    t->matmulTiling.usedCoreNum = usedCoreNum;
    t->matmulTiling.M = M;
    t->matmulTiling.N = N;
    t->matmulTiling.Ka = K;
    t->matmulTiling.Kb = K;
    t->matmulTiling.singleCoreM = singleCoreM;
    t->matmulTiling.singleCoreN = singleCoreN;
    t->matmulTiling.singleCoreK = singleCoreK;
    t->matmulTiling.baseM = baseM;
    t->matmulTiling.baseN = baseN;
    t->matmulTiling.baseK = baseK;
    t->matmulTiling.depthA1 = depthA1;
    t->matmulTiling.depthB1 = depthB1;
    t->matmulTiling.stepM = stepM;
    t->matmulTiling.stepN = stepN;
    t->matmulTiling.stepKa = stepKa;
    t->matmulTiling.stepKb = stepKb;
    t->matmulTiling.isBias = isBias;
    t->matmulTiling.transLength = transLength;
    t->matmulTiling.iterateOrder = iterateOrder;
    t->matmulTiling.dbL0A = dbL0A; 
    t->matmulTiling.dbL0B = dbL0B;
    t->matmulTiling.dbL0C = dbL0C;
    t->matmulTiling.depthAL1CacheUB = 0;
    t->matmulTiling.depthBL1CacheUB = 0;

    // tileL2cacheTiling
    t->tileL2cacheTiling.mTileCntL2 = mTileCntL2;
    t->tileL2cacheTiling.nTileCntL2 = nTileCntL2;
    t->tileL2cacheTiling.mTileBlock = mTileBlock;
    t->tileL2cacheTiling.nTileBlock = nTileBlock;
    t->tileL2cacheTiling.calOrder = calOrder;
    t->l2cacheUseInfo.l2CacheFlag = 0;

    // matmulRunInfo
    t->matmulRunInfo.transA = transA;
    t->matmulRunInfo.transB = transB;
    t->matmulRunInfo.nd2nzA = nd2nzA;
    t->matmulRunInfo.nd2nzB = nd2nzB;
    t->matmulRunInfo.isNzA = isNzA;
    t->matmulRunInfo.isNzB = isNzB;
    t->matmulRunInfo.isHf32 = isHf32;

    // тайлинг ND->NZ конверсии, читается только UnAligned-ядрами
    t->baseAN = baseAN;
    t->baseAD = baseAD;
    t->baseBN = baseBN;
    t->baseBD = baseBD;

    uint32_t blockDim = t->matmulTiling.usedCoreNum;

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *a = (uint8_t *)AscendC::GmAlloc(aSize);
    uint8_t *b = (uint8_t *)AscendC::GmAlloc(bSize);
    uint8_t *c = (uint8_t *)AscendC::GmAlloc(cSize);
    uint8_t *ws = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *offsetW = (uint8_t *)AscendC::GmAlloc(1024);

    memset(a, 0, aSize);
    memset(b, 0, bSize);
    ReadFile("./input/x1_gm.bin", aSize, a, aSize);
    ReadFile("./input/x2_gm.bin", bSize, b, bSize);
    memcpy(tiling, tilingHost, tilingSize);

    if (kernelType == 0) {
#ifdef MM_HAVE_BASE
        ICPU_RUN_KF(matmul_v3_base, blockDim, a, b, nullptr, offsetW, c, ws, tiling, nullptr);
#else
        fprintf(stderr, "[ERROR] base kernel not built (-t base | -t all)\n"); return 1;
#endif
    } else if (kernelType == 1) {
#ifdef MM_HAVE_SC_SPLITK
        AscendC::SetKernelMode(KernelMode::AIC_MODE);
        ICPU_RUN_KF(matmul_v3_singlecore_splitK, blockDim, a, b, nullptr, offsetW, c, ws, tiling, nullptr);
#else
        fprintf(stderr, "[ERROR] singlecoreSplitK not built\n"); return 1;
#endif
    } else if (kernelType == 2){
#ifdef MM_HAVE_DET_SPLITK
        ICPU_RUN_KF(matmul_v3_deterministic_splitK, blockDim, a, b, nullptr, offsetW, c, ws, tiling, nullptr);
#else
        fprintf(stderr, "[ERROR] deterministicSplitK not built\n"); return 1;
#endif
    }

    WriteFile("./output/output.bin", c, cSize);

    AscendC::GmFree((void *)a);
    AscendC::GmFree((void *)b);
    AscendC::GmFree((void *)c);
    AscendC::GmFree((void *)ws);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)offsetW);
#else
CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));

    if (EnvI("MM_PRINT_PLATFORM", 0)) {
        PrintPlatformInfo();
        CHECK_ACL(aclrtResetDevice(deviceId));
        CHECK_ACL(aclFinalize());
        free(tilingHost);
        return 0;
    }

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *aHost = nullptr, *bHost = nullptr, *cHost = nullptr;
    uint8_t *aDev = nullptr, *bDev = nullptr, *cDev = nullptr;
    uint8_t *wsDev = nullptr, *tilingDev = nullptr, *offsetWDev = nullptr;

    CHECK_ACL(aclrtMallocHost((void **)&aHost, aSize));
    CHECK_ACL(aclrtMallocHost((void **)&bHost, bSize));
    CHECK_ACL(aclrtMallocHost((void **)&cHost, cSize));
    CHECK_ACL(aclrtMalloc((void **)&aDev, aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&bDev, bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&cDev, cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&wsDev, sysWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&tilingDev, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&offsetWDev, 1024, ACL_MEM_MALLOC_HUGE_FIRST));

    memset(aHost, 0, aSize);
    memset(bHost, 0, bSize);
    ReadFile("./input/x1_gm.bin", aSize, aHost, aSize);
    ReadFile("./input/x2_gm.bin", bSize, bHost, bSize);

    CHECK_ACL(aclrtMemcpy(aDev, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(bDev, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(tilingDev, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Буфер под тики: MatmulV3Cycles::BUFFER_ELEMS слотов по uint32.
    // Без флага в ядро уходит nullptr, поэтому обычный прогон ничего не пишет.
    const uint32_t cycleElems = 2048;
    const size_t cycleBytes = cycleElems * sizeof(uint32_t);
    uint8_t *cycleDev = nullptr;
    uint32_t *cycleHost = nullptr;
    CHECK_ACL(aclrtMalloc((void **)&cycleDev, cycleBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost((void **)&cycleHost, cycleBytes));
    uint8_t *cyclePtr = cyclesMode ? cycleDev : nullptr;

    // Прогрев и повторы крутим внутри процесса: данные уже на устройстве,
    // а при kIndex == 0 ядро перезаписывает C, а не накапливает в него.
    const int32_t warmup = cyclesMode ? EnvI("MM_CYCLES_WARMUP", 1) : 0;
    const int32_t repeat = cyclesMode ? EnvI("MM_CYCLES_REPEAT", 3) : 1;
    const double cyclesPerUs = (double)EnvI("MM_CYCLES_PER_US", 50); // 50 МГц на 910B
    double bestUs = -1.0;

    for (int32_t iter = 0; iter < warmup + repeat; ++iter) {
        if (cyclesMode) {
            CHECK_ACL(aclrtMemset(cycleDev, cycleBytes, 0, cycleBytes));
        }

        if (kernelType == 0) {
#ifdef MM_HAVE_BASE
            ACLRT_LAUNCH_KERNEL(matmul_v3_base)
                (blockDim, stream, aDev, bDev, nullptr, offsetWDev, cDev, wsDev, tilingDev, cyclePtr);
#else
            fprintf(stderr, "[ERROR] base kernel not built\n"); return 1;
#endif
        } else if (kernelType == 1) {
#ifdef MM_HAVE_SC_SPLITK
            ACLRT_LAUNCH_KERNEL(matmul_v3_singlecore_splitK)
                (blockDim, stream, aDev, bDev, nullptr, offsetWDev, cDev, wsDev, tilingDev, cyclePtr);
#else
            fprintf(stderr, "[ERROR] singlecoreSplitK not built\n"); return 1;
#endif
        } else if (kernelType == 2){
#ifdef MM_HAVE_DET_SPLITK
            ACLRT_LAUNCH_KERNEL(matmul_v3_deterministic_splitK)
                (blockDim, stream, aDev, bDev, nullptr, offsetWDev, cDev, wsDev, tilingDev, cyclePtr);
#else
            fprintf(stderr, "[ERROR] deterministicSplitK not built\n"); return 1;
#endif
        }

        CHECK_ACL(aclrtSynchronizeStream(stream));

        if (!cyclesMode || iter < warmup) {
            continue;
        }

        CHECK_ACL(aclrtMemcpy(cycleHost, cycleBytes, cycleDev, cycleBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        // makespan ядра — максимум по слотам, а не время нулевого блока
        uint32_t maxTicks = 0;
        for (uint32_t i = 0; i < cycleElems; ++i) {
            if (cycleHost[i] > maxTicks) {
                maxTicks = cycleHost[i];
            }
        }
        const double us = maxTicks / cyclesPerUs;
        if (bestUs < 0.0 || us < bestUs) {
            bestUs = us;
        }
    }

    if (cyclesMode) {
        printf("CYCLES_US: %.3f\n", bestUs);
    }

    CHECK_ACL(aclrtMemcpy(cHost, cSize, cDev, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", cHost, cSize);

    CHECK_ACL(aclrtFree(cycleDev));
    CHECK_ACL(aclrtFreeHost(cycleHost));
    CHECK_ACL(aclrtFree(aDev));       
    CHECK_ACL(aclrtFree(bDev));
    CHECK_ACL(aclrtFree(cDev));       
    CHECK_ACL(aclrtFree(wsDev));
    CHECK_ACL(aclrtFree(tilingDev));  
    CHECK_ACL(aclrtFree(offsetWDev));
    CHECK_ACL(aclrtFreeHost(aHost));  
    CHECK_ACL(aclrtFreeHost(bHost));
    CHECK_ACL(aclrtFreeHost(cHost));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif

    free(tilingHost);
    return 0;
}