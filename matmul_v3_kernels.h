#ifndef MATMUL_V3_KERNELS_H
#define MATMUL_V3_KERNELS_H

// Объявления точек входа ядер из matmulV3_launch_*.cpp.
// В npu/sim заголовки aclrtlaunch_* генерирует ascendc_library,
// в cpu-режиме генерации нет — объявляем руками.
// Дефайны MM_HAVE_* приходят из CMake: собрано только выбранное ядро.

#include <cstdint>

#ifndef ASCENDC_CPU_DEBUG

#ifdef MM_HAVE_BASE
#include "aclrtlaunch_matmul_v3_base.h"
#endif
#ifdef MM_HAVE_SC_SPLITK
#include "aclrtlaunch_matmul_v3_singlecore_splitK.h"
#endif
#ifdef MM_HAVE_DET_SPLITK
#include "aclrtlaunch_matmul_v3_deterministic_splitK.h"
#endif

#else

#ifdef MM_HAVE_BASE
extern "C" void matmul_v3_base(uint8_t *a, uint8_t *b, uint8_t *bias,
                               uint8_t *offsetW, uint8_t *c,
                               uint8_t *workspace, uint8_t *tiling,
                               uint8_t *cycle);
#endif
#ifdef MM_HAVE_SC_SPLITK
extern "C" void matmul_v3_singlecore_splitK(uint8_t *a, uint8_t *b, uint8_t *bias,
                                            uint8_t *offsetW, uint8_t *c,
                                            uint8_t *workspace, uint8_t *tiling,
                                            uint8_t *cycle);
#endif
#ifdef MM_HAVE_DET_SPLITK
extern "C" void matmul_v3_deterministic_splitK(uint8_t *a, uint8_t *b, uint8_t *bias,
                                               uint8_t *offsetW, uint8_t *c,
                                               uint8_t *workspace, uint8_t *tiling,
                                               uint8_t *cycle);
#endif

#endif

#endif // MATMUL_V3_KERNELS_H
