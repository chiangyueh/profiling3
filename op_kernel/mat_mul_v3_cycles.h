#ifndef MAT_MUL_V3_CYCLES_H
#define MAT_MUL_V3_CYCLES_H

#include "kernel_operator.h"

// Замер времени ядра системным счётчиком. На 910B он тикает 50 МГц, то есть
// 20 нс на тик; при длительностях в десятки-сотни микросекунд это тысячи тиков,
// разрешения хватает. Пишем сырые тики — перевод в микросекунды делает хост,
// чтобы не зашивать частоту в ядро.
//
// Каждое ядро кладёт результат в свой слот с шагом SLOT_STRIDE, чтобы записи
// не попадали в одну строку кэша. Куб и вектор нумеруются независимо, поэтому
// вектор смещён в свою половину буфера — иначе они затирали бы друг друга.
namespace MatmulV3Cycles {

constexpr uint32_t SLOT_STRIDE = 16;                     // в uint32
constexpr uint32_t MAX_BLOCKS = 64;
constexpr uint32_t AIV_OFFSET = MAX_BLOCKS * SLOT_STRIDE;
constexpr uint32_t BUFFER_ELEMS = AIV_OFFSET * 2;        // 2048 uint32 = 8 КБ

__aicore__ inline void Write(GM_ADDR cycleGM, uint64_t ticks)
{
    // nullptr приходит в cpu-режиме и в обычных прогонах без MM_CYCLES
    if (cycleGM == nullptr) {
        return;
    }
    uint64_t blockIdx = AscendC::GetBlockIdx();
    if (blockIdx >= MAX_BLOCKS) {
        return;
    }
    uint32_t slot = static_cast<uint32_t>(blockIdx) * SLOT_STRIDE;
    if ASCEND_IS_AIV {
        slot += AIV_OFFSET;
    }
    __gm__ uint32_t *slots = reinterpret_cast<__gm__ uint32_t *>(cycleGM);
    slots[slot] = static_cast<uint32_t>(ticks);
}

}  // namespace MatmulV3Cycles

#endif  // MAT_MUL_V3_CYCLES_H
