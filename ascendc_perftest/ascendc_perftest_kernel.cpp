/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _MTEPERF_KERNEL_ASCENDC_PERFTEST_
#define _MTEPERF_KERNEL_ASCENDC_PERFTEST_

#include <cstring>
#include "kernel_operator.h"
#include "perftest_common_types.h"

template <typename T>
__aicore__ inline void rma_operation_ub2gm_remote_impl(GM_ADDR d1_ptr, GM_ADDR d0_ptr, GM_ADDR time_gva, 
                                                          int elements, int loop_count, int ub_size_kb)
{
#if !defined(__DAV_C310_VEC__) && !defined(__DAV_C310_CUBE__)
    int64_t cycle2us = 50;
#else
    int64_t cycle2us = 1000;
#endif
    __gm__ T *d1_gm = (__gm__ T *)d1_ptr;
    __gm__ T *d0_gm = (__gm__ T *)d0_ptr;
    __gm__ float *time_gm = (__gm__ float *)time_gva;

    uint64_t copy_ub = 0;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)0;

    int32_t tile_length = (int32_t)(elements / AscendC::GetBlockNum());
    uint64_t start_cycle, end_cycle;
    int32_t all_times = 0;
    int warmup = 100;
    int loop_test = loop_count;
    uint64_t block_size = tile_length * sizeof(T);

    int offset = (block_size < 512) ? 512 / sizeof(T) : block_size / sizeof(T);
    const uint64_t MAX_UB_SIZE = ub_size_kb * 1024;
    uint64_t actual_block_size = (block_size > MAX_UB_SIZE) ? MAX_UB_SIZE : block_size;
    uint64_t remain = block_size % actual_block_size;
    uint64_t repeat_times = block_size / actual_block_size;
    uint64_t repeat_elem = actual_block_size / sizeof(T);

    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> gm_tensor_remote;
    AscendC::DataCopyExtParams dataCopyParams;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);

    for (int i = 0; i < (warmup + loop_test); ++i) {
        pipe_barrier(PIPE_ALL);
        start_cycle = AscendC::GetSystemCycle();
        for (uint64_t j = 0; j < repeat_times; j++) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(copy_event_id);

            gm_tensor_remote.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d1_gm + offset * AscendC::GetBlockIdx() + j * repeat_elem));
            dataCopyParams = AscendC::DataCopyExtParams(1, actual_block_size, 0, 0, 0);
            AscendC::DataCopyPad(gm_tensor_remote, ub_tensor, dataCopyParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        }

        if (remain > 0) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(copy_event_id);

            gm_tensor_remote.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d1_gm + offset * AscendC::GetBlockIdx() + repeat_times * repeat_elem));
            dataCopyParams = AscendC::DataCopyExtParams(1, remain, 0, 0, 0);
            AscendC::DataCopyPad(gm_tensor_remote, ub_tensor, dataCopyParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        }
        pipe_barrier(PIPE_ALL);
        end_cycle = AscendC::GetSystemCycle();

        if (i >= warmup) {
            all_times += (int32_t)(end_cycle - start_cycle);
        }
    }

    float time_us1 = ((float)(all_times)) / (cycle2us) / loop_test;
    if (AscendC::GetBlockIdx() >= elements) {
        time_us1 = 0.0f;
    }
    *((__gm__ float *)time_gm + AscendC::GetBlockIdx() * 16) = (float)time_us1;
}

template <typename T>
__aicore__ inline void rma_operation_ascendc_put_impl(GM_ADDR d1_ptr, GM_ADDR d0_ptr, GM_ADDR time_gva,
                                                         int elements, int loop_count, int ub_size_kb)
{
#if !defined(__DAV_C310_VEC__) && !defined(__DAV_C310_CUBE__)
    int64_t cycle2us = 50;
#else
    int64_t cycle2us = 1000;
#endif
    __gm__ T *d1_gm = (__gm__ T *)d1_ptr;
    __gm__ T *d0_gm = (__gm__ T *)d0_ptr;
    __gm__ float *time_gm = (__gm__ float *)time_gva;

    uint64_t copy_ub = 0;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)0;

    int32_t tile_length = (int32_t)(elements / AscendC::GetBlockNum());
    uint64_t start_cycle, end_cycle;
    int32_t all_times = 0;
    int warmup = 100;
    int loop_test = loop_count;
    uint64_t block_size = tile_length * sizeof(T);

    int offset = (block_size < 512) ? 512 / sizeof(T) : block_size / sizeof(T);
    const uint64_t MAX_UB_SIZE = ub_size_kb * 1024;
    uint64_t actual_block_size = (block_size > MAX_UB_SIZE) ? MAX_UB_SIZE : block_size;
    uint64_t remain = block_size % actual_block_size;
    uint64_t repeat_times = block_size / actual_block_size;
    uint64_t repeat_elem = actual_block_size / sizeof(T);
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;

    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> gm_tensor_local;
    AscendC::GlobalTensor<T> gm_tensor_remote;
    AscendC::DataCopyExtParams dataCopyParams;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    AscendC::DataCopyPadExtParams<T> padParams;

    for (int i = 0; i < (warmup + loop_test); ++i) {
        pipe_barrier(PIPE_ALL);
        start_cycle = AscendC::GetSystemCycle();
        for (uint64_t j = 0; j < loop_times; j++) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(copy_event_id);

            if (j < repeat_times) {
                gm_tensor_local.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d0_gm + offset * AscendC::GetBlockIdx() + j * repeat_elem));
                gm_tensor_remote.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d1_gm + offset * AscendC::GetBlockIdx() + j * repeat_elem));
                dataCopyParams = AscendC::DataCopyExtParams(1, actual_block_size, 0, 0, 0);
                AscendC::DataCopyPad(ub_tensor, gm_tensor_local, dataCopyParams, padParams);
            } else {
                gm_tensor_local.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d0_gm + offset * AscendC::GetBlockIdx() + repeat_times * repeat_elem));
                gm_tensor_remote.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d1_gm + offset * AscendC::GetBlockIdx() + repeat_times * repeat_elem));
                dataCopyParams = AscendC::DataCopyExtParams(1, remain, 0, 0, 0);
                AscendC::DataCopyPad(ub_tensor, gm_tensor_local, dataCopyParams, padParams);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(copy_event_id);

            AscendC::DataCopyPad(gm_tensor_remote, ub_tensor, dataCopyParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copy_event_id);
        }
        pipe_barrier(PIPE_ALL);
        end_cycle = AscendC::GetSystemCycle();

        if (i >= warmup) {
            all_times += (int32_t)(end_cycle - start_cycle);
        }
    }

    float time_us1 = ((float)(all_times)) / (cycle2us) / loop_test;
    if (AscendC::GetBlockIdx() >= elements) {
        time_us1 = 0.0f;
    }
    *((__gm__ float *)time_gm + AscendC::GetBlockIdx() * 16) = (float)time_us1;
}

template <typename T>
__aicore__ inline void rma_operation_ascendc_get_impl(GM_ADDR d1_ptr, GM_ADDR d0_ptr, GM_ADDR time_gva,
                                                         int elements, int loop_count, int ub_size_kb)
{
#if !defined(__DAV_C310_VEC__) && !defined(__DAV_C310_CUBE__)
    int64_t cycle2us = 50;
#else
    int64_t cycle2us = 1000;
#endif
    __gm__ T *d1_gm = (__gm__ T *)d1_ptr;
    __gm__ T *d0_gm = (__gm__ T *)d0_ptr;
    __gm__ float *time_gm = (__gm__ float *)time_gva;

    uint64_t copy_ub = 0;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)0;

    int32_t tile_length = (int32_t)(elements / AscendC::GetBlockNum());
    uint64_t start_cycle, end_cycle;
    int32_t all_times = 0;
    int warmup = 100;
    int loop_test = loop_count;
    uint64_t block_size = tile_length * sizeof(T);

    int offset = (block_size < 512) ? 512 / sizeof(T) : block_size / sizeof(T);
    const uint64_t MAX_UB_SIZE = ub_size_kb * 1024;
    uint64_t actual_block_size = (block_size > MAX_UB_SIZE) ? MAX_UB_SIZE : block_size;
    uint64_t remain = block_size % actual_block_size;
    uint64_t repeat_times = block_size / actual_block_size;
    uint64_t repeat_elem = actual_block_size / sizeof(T);
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;

    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> gm_tensor_local;
    AscendC::GlobalTensor<T> gm_tensor_remote;
    AscendC::DataCopyExtParams dataCopyParams;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    AscendC::DataCopyPadExtParams<T> padParams;

    for (int i = 0; i < (warmup + loop_test); ++i) {
        pipe_barrier(PIPE_ALL);
        start_cycle = AscendC::GetSystemCycle();
        for (uint64_t j = 0; j < loop_times; j++) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(copy_event_id);

            if (j < repeat_times) {
                gm_tensor_remote.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d1_gm + offset * AscendC::GetBlockIdx() + j * repeat_elem));
                gm_tensor_local.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d0_gm + offset * AscendC::GetBlockIdx() + j * repeat_elem));
                dataCopyParams = AscendC::DataCopyExtParams(1, actual_block_size, 0, 0, 0);
                AscendC::DataCopyPad(ub_tensor, gm_tensor_remote, dataCopyParams, padParams);
            } else {
                gm_tensor_remote.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d1_gm + offset * AscendC::GetBlockIdx() + repeat_times * repeat_elem));
                gm_tensor_local.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d0_gm + offset * AscendC::GetBlockIdx() + repeat_times * repeat_elem));
                dataCopyParams = AscendC::DataCopyExtParams(1, remain, 0, 0, 0);
                AscendC::DataCopyPad(ub_tensor, gm_tensor_remote, dataCopyParams, padParams);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(copy_event_id);

            AscendC::DataCopyPad(gm_tensor_local, ub_tensor, dataCopyParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copy_event_id);
        }
        pipe_barrier(PIPE_ALL);
        end_cycle = AscendC::GetSystemCycle();

        if (i >= warmup) {
            all_times += (int32_t)(end_cycle - start_cycle);
        }
    }

    float time_us1 = ((float)(all_times)) / (cycle2us) / loop_test;
    if (AscendC::GetBlockIdx() >= elements) {
        time_us1 = 0.0f;
    }
    *((__gm__ float *)time_gm + AscendC::GetBlockIdx() * 16) = (float)time_us1;
}

template <typename T>
__aicore__ inline void rma_operation_ub2gm_local_impl(GM_ADDR d1_ptr, GM_ADDR d0_ptr, GM_ADDR time_gva,
                                                         int elements, int loop_count, int ub_size_kb)
{
#if !defined(__DAV_C310_VEC__) && !defined(__DAV_C310_CUBE__)
    int64_t cycle2us = 50;
#else
    int64_t cycle2us = 1000;
#endif
    __gm__ T *d1_gm = (__gm__ T *)d1_ptr;
    __gm__ T *d0_gm = (__gm__ T *)d0_ptr;
    __gm__ float *time_gm = (__gm__ float *)time_gva;

    uint64_t copy_ub = 0;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)0;

    int32_t tile_length = (int32_t)(elements / AscendC::GetBlockNum());
    uint64_t start_cycle, end_cycle;
    int32_t all_times = 0;
    int warmup = 100;
    int loop_test = loop_count;
    uint64_t block_size = tile_length * sizeof(T);

    int offset = (block_size < 512) ? 512 / sizeof(T) : block_size / sizeof(T);
    const uint64_t MAX_UB_SIZE = ub_size_kb * 1024;
    uint64_t actual_block_size = (block_size > MAX_UB_SIZE) ? MAX_UB_SIZE : block_size;
    uint64_t remain = block_size % actual_block_size;
    uint64_t repeat_times = block_size / actual_block_size;
    uint64_t repeat_elem = actual_block_size / sizeof(T);

    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> gm_tensor_local;
    AscendC::DataCopyExtParams dataCopyParams;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);

    for (int i = 0; i < (warmup + loop_test); ++i) {
        pipe_barrier(PIPE_ALL);
        start_cycle = AscendC::GetSystemCycle();
        for (uint64_t j = 0; j < repeat_times; j++) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(copy_event_id);

            gm_tensor_local.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d0_gm + offset * AscendC::GetBlockIdx() + j * repeat_elem));
            dataCopyParams = AscendC::DataCopyExtParams(1, actual_block_size, 0, 0, 0);
            AscendC::DataCopyPad(gm_tensor_local, ub_tensor, dataCopyParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        }

        if (remain > 0) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(copy_event_id);

            gm_tensor_local.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d0_gm + offset * AscendC::GetBlockIdx() + repeat_times * repeat_elem));
            dataCopyParams = AscendC::DataCopyExtParams(1, remain, 0, 0, 0);
            AscendC::DataCopyPad(gm_tensor_local, ub_tensor, dataCopyParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copy_event_id);
        }
        pipe_barrier(PIPE_ALL);
        end_cycle = AscendC::GetSystemCycle();

        if (i >= warmup) {
            all_times += (int32_t)(end_cycle - start_cycle);
        }
    }

    float time_us1 = ((float)(all_times)) / (cycle2us) / loop_test;
    if (AscendC::GetBlockIdx() >= elements) {
        time_us1 = 0.0f;
    }
    *((__gm__ float *)time_gm + AscendC::GetBlockIdx() * 16) = (float)time_us1;
}

template <typename T>
__aicore__ inline void rma_operation_gm2ub_local_impl(GM_ADDR d1_ptr, GM_ADDR d0_ptr, GM_ADDR time_gva,
                                                         int elements, int loop_count, int ub_size_kb)
{
#if !defined(__DAV_C310_VEC__) && !defined(__DAV_C310_CUBE__)
    int64_t cycle2us = 50;
#else
    int64_t cycle2us = 1000;
#endif
    __gm__ T *d1_gm = (__gm__ T *)d1_ptr;
    __gm__ T *d0_gm = (__gm__ T *)d0_ptr;
    __gm__ float *time_gm = (__gm__ float *)time_gva;

    uint64_t copy_ub = 0;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)0;

    int32_t tile_length = (int32_t)(elements / AscendC::GetBlockNum());
    uint64_t start_cycle, end_cycle;
    int32_t all_times = 0;
    int warmup = 100;
    int loop_test = loop_count;
    uint64_t block_size = tile_length * sizeof(T);

    int offset = (block_size < 512) ? 512 / sizeof(T) : block_size / sizeof(T);
    const uint64_t MAX_UB_SIZE = ub_size_kb * 1024;
    uint64_t actual_block_size = (block_size > MAX_UB_SIZE) ? MAX_UB_SIZE : block_size;
    uint64_t remain = block_size % actual_block_size;
    uint64_t repeat_times = block_size / actual_block_size;
    uint64_t repeat_elem = actual_block_size / sizeof(T);

    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> gm_tensor_local;
    AscendC::DataCopyExtParams dataCopyParams;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    AscendC::DataCopyPadExtParams<T> padParams;

    for (int i = 0; i < (warmup + loop_test); ++i) {
        pipe_barrier(PIPE_ALL);
        start_cycle = AscendC::GetSystemCycle();
        for (uint64_t j = 0; j < repeat_times; j++) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(copy_event_id);

            gm_tensor_local.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d0_gm + offset * AscendC::GetBlockIdx() + j * repeat_elem));
            dataCopyParams = AscendC::DataCopyExtParams(1, actual_block_size, 0, 0, 0);
            AscendC::DataCopyPad(ub_tensor, gm_tensor_local, dataCopyParams, padParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(copy_event_id);
        }

        if (remain > 0) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(copy_event_id);

            gm_tensor_local.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d0_gm + offset * AscendC::GetBlockIdx() + repeat_times * repeat_elem));
            dataCopyParams = AscendC::DataCopyExtParams(1, remain, 0, 0, 0);
            AscendC::DataCopyPad(ub_tensor, gm_tensor_local, dataCopyParams, padParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(copy_event_id);
        }
        pipe_barrier(PIPE_ALL);
        end_cycle = AscendC::GetSystemCycle();

        if (i >= warmup) {
            all_times += (int32_t)(end_cycle - start_cycle);
        }
    }

    float time_us1 = ((float)(all_times)) / (cycle2us) / loop_test;
    if (AscendC::GetBlockIdx() >= elements) {
        time_us1 = 0.0f;
    }
    *((__gm__ float *)time_gm + AscendC::GetBlockIdx() * 16) = (float)time_us1;
}

template <typename T>
__aicore__ inline void rma_operation_gm2ub_remote_impl(GM_ADDR d1_ptr, GM_ADDR d0_ptr, GM_ADDR time_gva,
                                                          int elements, int loop_count, int ub_size_kb)
{
#if !defined(__DAV_C310_VEC__) && !defined(__DAV_C310_CUBE__)
    int64_t cycle2us = 50;
#else
    int64_t cycle2us = 1000;
#endif
    __gm__ T *d1_gm = (__gm__ T *)d1_ptr;
    __gm__ T *d0_gm = (__gm__ T *)d0_ptr;
    __gm__ float *time_gm = (__gm__ float *)time_gva;

    uint64_t copy_ub = 0;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)0;

    int32_t tile_length = (int32_t)(elements / AscendC::GetBlockNum());
    uint64_t start_cycle, end_cycle;
    int32_t all_times = 0;
    int warmup = 100;
    int loop_test = loop_count;
    uint64_t block_size = tile_length * sizeof(T);

    int offset = (block_size < 512) ? 512 / sizeof(T) : block_size / sizeof(T);
    const uint64_t MAX_UB_SIZE = ub_size_kb * 1024;
    uint64_t actual_block_size = (block_size > MAX_UB_SIZE) ? MAX_UB_SIZE : block_size;
    uint64_t remain = block_size % actual_block_size;
    uint64_t repeat_times = block_size / actual_block_size;
    uint64_t repeat_elem = actual_block_size / sizeof(T);

    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> gm_tensor_remote;
    AscendC::DataCopyExtParams dataCopyParams;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    AscendC::DataCopyPadExtParams<T> padParams;

    for (int i = 0; i < (warmup + loop_test); ++i) {
        pipe_barrier(PIPE_ALL);
        start_cycle = AscendC::GetSystemCycle();
        for (uint64_t j = 0; j < repeat_times; j++) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(copy_event_id);

            gm_tensor_remote.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d1_gm + offset * AscendC::GetBlockIdx() + j * repeat_elem));
            dataCopyParams = AscendC::DataCopyExtParams(1, actual_block_size, 0, 0, 0);
            AscendC::DataCopyPad(ub_tensor, gm_tensor_remote, dataCopyParams, padParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(copy_event_id);
        }

        if (remain > 0) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(copy_event_id);

            gm_tensor_remote.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d1_gm + offset * AscendC::GetBlockIdx() + repeat_times * repeat_elem));
            dataCopyParams = AscendC::DataCopyExtParams(1, remain, 0, 0, 0);
            AscendC::DataCopyPad(ub_tensor, gm_tensor_remote, dataCopyParams, padParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(copy_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(copy_event_id);
        }
        pipe_barrier(PIPE_ALL);
        end_cycle = AscendC::GetSystemCycle();

        if (i >= warmup) {
            all_times += (int32_t)(end_cycle - start_cycle);
        }
    }

    float time_us1 = ((float)(all_times)) / (cycle2us) / loop_test;
    if (AscendC::GetBlockIdx() >= elements) {
        time_us1 = 0.0f;
    }
    *((__gm__ float *)time_gm + AscendC::GetBlockIdx() * 16) = (float)time_us1;
}

#define DEFINE_RMA_OPERATION_KERNEL_FOR_TYPE(type_name, cpp_type) \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rma_operation_##type_name##_ub2gm_remote(GM_ADDR d1_ptr, GM_ADDR d0_ptr, GM_ADDR time_gva, int elements, int loop_count, int ub_size_kb) \
{ \
    rma_operation_ub2gm_remote_impl<cpp_type>(d1_ptr, d0_ptr, time_gva, elements, loop_count, ub_size_kb); \
} \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rma_operation_##type_name##_ascendc_put(GM_ADDR d1_ptr, GM_ADDR d0_ptr, GM_ADDR time_gva, int elements, int loop_count, int ub_size_kb) \
{ \
    rma_operation_ascendc_put_impl<cpp_type>(d1_ptr, d0_ptr, time_gva, elements, loop_count, ub_size_kb); \
} \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rma_operation_##type_name##_ascendc_get(GM_ADDR d1_ptr, GM_ADDR d0_ptr, GM_ADDR time_gva, int elements, int loop_count, int ub_size_kb) \
{ \
    rma_operation_ascendc_get_impl<cpp_type>(d1_ptr, d0_ptr, time_gva, elements, loop_count, ub_size_kb); \
} \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rma_operation_##type_name##_ub2gm_local(GM_ADDR d1_ptr, GM_ADDR d0_ptr, GM_ADDR time_gva, int elements, int loop_count, int ub_size_kb) \
{ \
    rma_operation_ub2gm_local_impl<cpp_type>(d1_ptr, d0_ptr, time_gva, elements, loop_count, ub_size_kb); \
} \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rma_operation_##type_name##_gm2ub_local(GM_ADDR d1_ptr, GM_ADDR d0_ptr, GM_ADDR time_gva, int elements, int loop_count, int ub_size_kb) \
{ \
    rma_operation_gm2ub_local_impl<cpp_type>(d1_ptr, d0_ptr, time_gva, elements, loop_count, ub_size_kb); \
} \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rma_operation_##type_name##_gm2ub_remote(GM_ADDR d1_ptr, GM_ADDR d0_ptr, GM_ADDR time_gva, int elements, int loop_count, int ub_size_kb) \
{ \
    rma_operation_gm2ub_remote_impl<cpp_type>(d1_ptr, d0_ptr, time_gva, elements, loop_count, ub_size_kb); \
}

DEFINE_RMA_OPERATION_KERNEL_FOR_TYPE(float, float)
DEFINE_RMA_OPERATION_KERNEL_FOR_TYPE(int8, int8_t)
DEFINE_RMA_OPERATION_KERNEL_FOR_TYPE(int16, int16_t)
DEFINE_RMA_OPERATION_KERNEL_FOR_TYPE(int32, int32_t)
DEFINE_RMA_OPERATION_KERNEL_FOR_TYPE(int64, int64_t)
DEFINE_RMA_OPERATION_KERNEL_FOR_TYPE(uint8, uint8_t)
DEFINE_RMA_OPERATION_KERNEL_FOR_TYPE(uint16, uint16_t)
DEFINE_RMA_OPERATION_KERNEL_FOR_TYPE(uint32, uint32_t)
DEFINE_RMA_OPERATION_KERNEL_FOR_TYPE(uint64, uint64_t)
DEFINE_RMA_OPERATION_KERNEL_FOR_TYPE(char, char)

#define DISPATCH_RMA_OPERATION_FOR_TYPE_AND_TEST(type_name, cpp_type, test_mode) \
    rma_operation_##type_name##_##test_mode<<<block_dim, nullptr, stream>>>(d1_ptr, d0_ptr, time_gva, elements, loop_count, ub_size_kb)

#define DISPATCH_RMA_OPERATION_FOR_ALL_TYPES(test_mode) \
    switch (d_type) { \
        case perftest::DATA_TYPE_FLOAT: \
            DISPATCH_RMA_OPERATION_FOR_TYPE_AND_TEST(float, float, test_mode); \
            break; \
        case perftest::DATA_TYPE_INT8: \
            DISPATCH_RMA_OPERATION_FOR_TYPE_AND_TEST(int8, int8_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_INT16: \
            DISPATCH_RMA_OPERATION_FOR_TYPE_AND_TEST(int16, int16_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_INT32: \
            DISPATCH_RMA_OPERATION_FOR_TYPE_AND_TEST(int32, int32_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_INT64: \
            DISPATCH_RMA_OPERATION_FOR_TYPE_AND_TEST(int64, int64_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_UINT8: \
            DISPATCH_RMA_OPERATION_FOR_TYPE_AND_TEST(uint8, uint8_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_UINT16: \
            DISPATCH_RMA_OPERATION_FOR_TYPE_AND_TEST(uint16, uint16_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_UINT32: \
            DISPATCH_RMA_OPERATION_FOR_TYPE_AND_TEST(uint32, uint32_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_UINT64: \
            DISPATCH_RMA_OPERATION_FOR_TYPE_AND_TEST(uint64, uint64_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_CHAR: \
            DISPATCH_RMA_OPERATION_FOR_TYPE_AND_TEST(char, char, test_mode); \
            break; \
        default: \
            DISPATCH_RMA_OPERATION_FOR_TYPE_AND_TEST(float, float, test_mode); \
            break; \
    }

extern "C" void device_perftest_demo(uint32_t block_dim, void *stream, uint8_t *d1_ptr, uint8_t *d0_ptr, uint8_t *time_gva, int elements, int test_type, int loop_count, int data_type, int ub_size_kb)
{
    perftest::ascendc_mode_t t_type = static_cast<perftest::ascendc_mode_t>(test_type);
    perftest::perf_data_type_t d_type = static_cast<perftest::perf_data_type_t>(data_type);

    switch (t_type) {
        case perftest::TEST_MODE_UB2GM_REMOTE:
            DISPATCH_RMA_OPERATION_FOR_ALL_TYPES(ub2gm_remote);
            break;
        case perftest::TEST_MODE_ASCENDC_PUT:
            DISPATCH_RMA_OPERATION_FOR_ALL_TYPES(ascendc_put);
            break;
        case perftest::TEST_MODE_ASCENDC_GET:
            DISPATCH_RMA_OPERATION_FOR_ALL_TYPES(ascendc_get);
            break;
        case perftest::TEST_MODE_UB2GM_LOCAL:
            DISPATCH_RMA_OPERATION_FOR_ALL_TYPES(ub2gm_local);
            break;
        case perftest::TEST_MODE_GM2UB_LOCAL:
            DISPATCH_RMA_OPERATION_FOR_ALL_TYPES(gm2ub_local);
            break;
        case perftest::TEST_MODE_GM2UB_REMOTE:
            DISPATCH_RMA_OPERATION_FOR_ALL_TYPES(gm2ub_remote);
            break;
        default:
            break;
    }
}

#endif  // _MTEPERF_KERNEL_ASCENDC_PERFTEST_
