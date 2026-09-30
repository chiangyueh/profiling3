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
 * \file flash_attention_score_grad_tiling_data_regbase.h
 * \brief
 */

 #ifndef FLASH_ATTENTION_SCORE_GRAD_TILING_DATA_REGBASE_H_
 #define FLASH_ATTENTION_SCORE_GRAD_TILING_DATA_REGBASE_H_

 #include <cstdint>
 
 namespace optiling {
 namespace fag {
 constexpr uint32_t MAX_CORE_NUM = 36;
 class FlashAttentionScoreGradEmptyTensorTilingDataRegbase {
 public:
     uint32_t formerDqNum;
     uint32_t formerDkNum;
     uint32_t formerDvNum;
     uint32_t formerDpseNum;
     uint64_t singleCoreDqNum;
     uint64_t tailCoreDqNum;
     uint64_t singleCoreDkNum;
     uint64_t tailCoreDkNum;
     uint64_t singleCoreDvNum;
     uint64_t tailCoreDvNum;
     uint64_t singleCoreDpseNum;
     uint64_t tailCoreDpseNum;
 
     uint32_t get_formerDqNum() const { return formerDqNum; }
     uint32_t get_formerDkNum() const { return formerDkNum; }
     uint32_t get_formerDvNum() const { return formerDvNum; }
     uint32_t get_formerDpseNum() const { return formerDpseNum; }
     uint64_t get_singleCoreDqNum() const { return singleCoreDqNum; }
     uint64_t get_tailCoreDqNum() const { return tailCoreDqNum; }
     uint64_t get_singleCoreDkNum() const { return singleCoreDkNum; }
     uint64_t get_tailCoreDkNum() const { return tailCoreDkNum; }
     uint64_t get_singleCoreDvNum() const { return singleCoreDvNum; }
     uint64_t get_tailCoreDvNum() const { return tailCoreDvNum; }
     uint64_t get_singleCoreDpseNum() const { return singleCoreDpseNum; }
     uint64_t get_tailCoreDpseNum() const { return tailCoreDpseNum; }
 
     void set_formerDqNum(uint32_t formerDqNumParam) { this->formerDqNum = formerDqNumParam; }
     void set_formerDkNum(uint32_t formerDkNumParam) { this->formerDkNum = formerDkNumParam; }
     void set_formerDvNum(uint32_t formerDvNumParam) { this->formerDvNum = formerDvNumParam; }
     void set_formerDpseNum(uint32_t formerDpseNumParam) { this->formerDpseNum = formerDpseNumParam; }
     void set_singleCoreDqNum(uint64_t singleCoreDqNumParam) { this->singleCoreDqNum = singleCoreDqNumParam; }
     void set_tailCoreDqNum(uint64_t tailCoreDqNumParam) { this->tailCoreDqNum = tailCoreDqNumParam; }
     void set_singleCoreDkNum(uint64_t singleCoreDkNumParam) { this->singleCoreDkNum = singleCoreDkNumParam; }
     void set_tailCoreDkNum(uint64_t tailCoreDkNumParam) { this->tailCoreDkNum = tailCoreDkNumParam; }
     void set_singleCoreDvNum(uint64_t singleCoreDvNumParam) { this->singleCoreDvNum = singleCoreDvNumParam; }
     void set_tailCoreDvNum(uint64_t tailCoreDvNumParam) { this->tailCoreDvNum = tailCoreDvNumParam; }
     void set_singleCoreDpseNum(uint64_t singleCoreDpseNumParam) { this->singleCoreDpseNum = singleCoreDpseNumParam; }
     void set_tailCoreDpseNum(uint64_t tailCoreDpseNumParam) { this->tailCoreDpseNum = tailCoreDpseNumParam; }
 };
 
 class FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase {
 public:
     int64_t coreNum;
     int64_t b;
     int64_t n2;
     int64_t g;
     int64_t s1;
     int64_t s2;
     int64_t d;
     int64_t d1;
     float scaleValue;
     float keepProb;
     int64_t keepProbUint8;
     int64_t s1Token;
     int64_t s2Token;
     int64_t seed;
     int64_t offset;
     int64_t totalPerBatchNum;
     uint32_t layout;
     uint32_t pseOptional;
     uint32_t pseType;
     uint32_t pseShapeType;
     uint32_t pseLayoutType;
     uint32_t pseDtype;
     uint32_t qStartIdx;
     uint32_t kvStartIdx;
     uint32_t attenMaskOptional;
     uint32_t attenMaskDtype;
     uint32_t attenMaskShapeType;
     uint32_t sparseMode;
     uint32_t attenMaskCompressMode;
     uint32_t attenMaskS2Size;
     uint32_t reserved1;    // tilingData需要8字节对齐
     uint8_t isSplitByBlockIdx;
     uint8_t dropMaskOuter;
     uint8_t sparseType;
     uint8_t reserved2;  // tilingData需要8字节对齐
 
     int64_t get_coreNum() const {return coreNum;}
     int64_t get_b() const {return b;}
     int64_t get_n2() const {return n2;}
     int64_t get_g() const {return g;}
     int64_t get_s1() const {return s1;}
     int64_t get_s2() const {return s2;}
     int64_t get_d() const {return d;}
     int64_t get_d1() const {return d1;}
     float get_scaleValue() const {return scaleValue;}
     float get_keepProb() const {return keepProb;}
     int64_t get_keepProbUint8() const {return keepProbUint8;}
     uint8_t get_dropMaskOuter() const {return dropMaskOuter;}
     uint32_t get_layout() const {return layout;}
     uint32_t get_pseOptional() const {return pseOptional;}
     uint32_t get_pseType() const {return pseType;}
     uint32_t get_qStartIdx() const {return qStartIdx;}
     uint32_t get_kvStartIdx() const {return kvStartIdx;}
     uint32_t get_attenMaskOptional() const {return attenMaskOptional;}
     uint32_t get_attenMaskDtype() const {return attenMaskDtype;}
     uint32_t get_attenMaskShapeType() const {return attenMaskShapeType;}
     int64_t get_s1Token() const {return s1Token;}
     int64_t get_s2Token() const {return s2Token;}
     uint32_t get_sparseMode() const {return sparseMode;}
     int64_t get_seed() const {return seed;}
     int64_t get_offset() const {return offset;}
     uint32_t get_attenMaskCompressMode() const {return attenMaskCompressMode;}
     uint32_t get_attenMaskS2Size() const {return attenMaskS2Size;}
     uint8_t get_isSplitByBlockIdx() const {return isSplitByBlockIdx;}
     int64_t get_totalPerBatchNum() const {return totalPerBatchNum;}
     uint8_t get_sparseType() const {return sparseType;}
 
     void set_coreNum(int64_t coreNumParam) { this->coreNum = coreNumParam; }
     void set_b(int64_t bParam) { this->b = bParam; }
     void set_n2(int64_t n2Param) { this->n2 = n2Param; }
     void set_g(int64_t gParam) { this->g = gParam; }
     void set_s1(int64_t s1Param) { this->s1 = s1Param; }
     void set_s2(int64_t s2Param) { this->s2 = s2Param; }
     void set_d(int64_t dParam) { this->d = dParam; }
     void set_d1(int64_t d1Param) { this->d1 = d1Param; }
     void set_scaleValue(float scaleValueParam) { this->scaleValue = scaleValueParam; }
     void set_keepProb(float keepProbParam) { this->keepProb = keepProbParam; }
     void set_keepProbUint8(int64_t keepProbUint8Param) { this->keepProbUint8 = keepProbUint8Param; }
     void set_dropMaskOuter(uint8_t dropMaskOuterParam) { this->dropMaskOuter = dropMaskOuterParam; }
     void set_layout(uint32_t layoutParam) { this->layout = layoutParam; }
     void set_pseOptional(uint32_t pseOptionalParam) { this->pseOptional = pseOptionalParam; }
     void set_pseType(uint32_t pseTypeParam) { this->pseType = pseTypeParam; }
     void set_pseShapeType(uint32_t pseShapeTypeParam) { this->pseShapeType = pseShapeTypeParam; }
     void set_pseLayoutType(uint32_t pseLayoutTypeParam) { this->pseLayoutType = pseLayoutTypeParam; }
     void set_pseDtype(uint32_t pseDtypeParam) { this->pseDtype = pseDtypeParam; }
     void set_qStartIdx(uint32_t qStartIdxParam) { this->qStartIdx = qStartIdxParam; }
     void set_kvStartIdx(uint32_t kvStartIdxParam) { this->kvStartIdx = kvStartIdxParam; }
     void set_attenMaskOptional(uint32_t attenMaskOptionalParam) { this->attenMaskOptional = attenMaskOptionalParam; }
     void set_attenMaskDtype(uint32_t attenMaskDtypeParam) { this->attenMaskDtype = attenMaskDtypeParam; }
     void set_attenMaskShapeType(uint32_t attenMaskShapeTypeParam) { this->attenMaskShapeType = attenMaskShapeTypeParam; }
     void set_s1Token(int64_t s1TokenParam) { this->s1Token = s1TokenParam; }
     void set_s2Token(int64_t s2TokenParam) { this->s2Token = s2TokenParam; }
     void set_sparseMode(uint32_t sparseModeParam) { this->sparseMode = sparseModeParam; }
     void set_seed(int64_t seedParam) { this->seed = seedParam; }
     void set_offset(int64_t offsetParam) { this->offset = offsetParam; }
     void set_attenMaskCompressMode(uint32_t attenMaskCompressModeParam) { this->attenMaskCompressMode = attenMaskCompressModeParam; }
     void set_attenMaskS2Size(uint32_t attenMaskS2SizeParam) { this->attenMaskS2Size = attenMaskS2SizeParam; }
     void set_isSplitByBlockIdx(uint8_t isSplitByBlockIdxParam) { this->isSplitByBlockIdx = isSplitByBlockIdxParam; }
     void set_totalPerBatchNum(int64_t totalPerBatchNumParam) { this->totalPerBatchNum = totalPerBatchNumParam; }
     void set_sparseType(uint8_t sparseTypeParam) { this->sparseType = sparseTypeParam; }
 };
 
 class FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParamsRegbase {
 public:
     int64_t s1Outer;
     uint32_t s1Inner;
     uint32_t s1CvInner;
     uint32_t s1Tail;
     uint32_t s1CvTail;
     int64_t s2Outer;
     uint32_t s2Inner;
     uint32_t s2Tail;
     uint32_t blockOuter;
     uint32_t maxValidBBLen;
     uint32_t noNeedDeter;
     uint32_t reserved1; // tilingData需要8字节对齐
     int64_t bandIdx;
     int64_t deterMaxRound;
     uint64_t dqIsNeedDeter[MAX_CORE_NUM];
     uint64_t dkDvIsNeedDeter[MAX_CORE_NUM];
 
     int64_t get_s1Outer() const { return s1Outer; }
     uint32_t get_s1Inner() const { return s1Inner; }
     uint32_t get_s1CvInner() const { return s1CvInner; }
     uint32_t get_s1Tail() const { return s1Tail; }
     uint32_t get_s1CvTail() const { return s1CvTail; }
     int64_t get_s2Outer() const { return s2Outer; }
     uint32_t get_s2Inner() const { return s2Inner; }
     uint32_t get_s2Tail() const { return s2Tail; }
     uint32_t get_blockOuter() const { return blockOuter; }
     uint32_t get_maxValidBBLen() const { return maxValidBBLen; }
     uint32_t get_noNeedDeter() const { return noNeedDeter; }
     int64_t get_bandIdx() const { return bandIdx; }
     int64_t get_deterMaxRound() const { return deterMaxRound; }
     const uint64_t* get_dqIsNeedDeter() const { return dqIsNeedDeter; }
     uint64_t get_dqIsNeedDeter(int index) const { return dqIsNeedDeter[index]; }
     const uint64_t* get_dkDvIsNeedDeter() const { return dkDvIsNeedDeter; }
     uint64_t get_dkDvIsNeedDeter(int index) const { return dkDvIsNeedDeter[index]; }
 
     void set_s1Outer(int64_t val) { s1Outer = val; }
     void set_s1Inner(uint32_t val) { s1Inner = val; }
     void set_s1CvInner(uint32_t val) { s1CvInner = val; }
     void set_s1Tail(uint32_t val) { s1Tail = val; }
     void set_s1CvTail(uint32_t val) { s1CvTail = val; }
     void set_s2Outer(int64_t val) { s2Outer = val; }
     void set_s2Inner(uint32_t val) { s2Inner = val; }
     void set_s2Tail(uint32_t val) { s2Tail = val; }
     void set_blockOuter(uint32_t val) { blockOuter = val; }
     void set_maxValidBBLen(uint32_t val) { maxValidBBLen = val; }
     void set_noNeedDeter(uint32_t val) { noNeedDeter = val; }
     void set_bandIdx(int64_t val) { bandIdx = val; }
     void set_deterMaxRound(int64_t value) { deterMaxRound = value; }
     void set_dqIsNeedDeter(const uint64_t* val) { 
         for (int i = 0; i < MAX_CORE_NUM; ++i) {
             dqIsNeedDeter[i] = val[i];
         }
     }
     void set_dqIsNeedDeter(int index, uint64_t val) { dqIsNeedDeter[index] = val; }
     void set_dkDvIsNeedDeter(const uint64_t* val) { 
         for (int i = 0; i < MAX_CORE_NUM; ++i) {
             dkDvIsNeedDeter[i] = val[i];
         }
     }
     void set_dkDvIsNeedDeter(int index, uint64_t val) { dkDvIsNeedDeter[index] = val; }
 };
 
 class BlockNumListParamsRegbase {
 public:
     int64_t blockStarts[MAX_CORE_NUM];
     int64_t blockEnds[MAX_CORE_NUM];
 
     const int64_t* get_blockStarts() const { return blockStarts;}
     int64_t get_blockStarts(int index) const { return blockStarts[index]; }
     const int64_t* get_blockEnds() const { return blockEnds; }
     int64_t get_blockEnds(int index) const { return blockEnds[index]; }
     void set_blockStarts(const int64_t* val) {
         for (int i = 0; i < MAX_CORE_NUM; ++i) {
             blockStarts[i] = val[i];
         }
     }
     void set_blockStarts(int index, int64_t val) { blockStarts[index] = val; }
     void set_blockEnds(const int64_t* val) {
         for (int i = 0; i < MAX_CORE_NUM; ++i) {
             blockEnds[i] = val[i];
         }
     }
     void set_blockEnds(int index, int64_t val) { blockEnds[index] = val; }
 };
 
 class PreParamsRegbase {
 public:
     uint64_t maskPreBlockTotal;
     uint64_t maskSingleCoreNum;
     uint32_t qPreBlockFactor;
     uint64_t qPreBlockTotal;
     uint32_t qPreBlockTail;
     uint32_t kPreBlockFactor;
     uint64_t kPreBlockTotal;
     uint32_t kPreBlockTail;
     uint32_t vPreBlockFactor;
     uint64_t vPreBlockTotal;
     uint32_t vPreBlockTail;
     uint32_t maskCoreNum;
     uint32_t castBufferLen;
     uint32_t outputBufferLen;
     uint32_t inputBufferLen;
     uint32_t singleUBProcessNum;
     uint32_t maskSingleCoreLoop;
     uint32_t maskLastLoopNum;
     uint32_t maskTailCoreLoop;
     uint32_t maskTailCoreLastLoopNum;
     uint32_t dropoutIsDivisibleBy8;
     bool sValueZeroUnderTND;
     uint8_t reserved1; // tilingData需要8字节对齐
     uint8_t reserved2; // tilingData需要8字节对齐
     uint8_t reserved3; // tilingData需要8字节对齐
     uint32_t reserved4; // tilingData需要8字节对齐
 
     uint64_t get_maskPreBlockTotal() const { return maskPreBlockTotal; }
     uint64_t get_maskSingleCoreNum() const { return maskSingleCoreNum; }
     uint32_t get_qPreBlockFactor() const { return qPreBlockFactor; }
     uint64_t get_qPreBlockTotal() const { return qPreBlockTotal; }
     uint32_t get_qPreBlockTail() const { return qPreBlockTail; }
     uint32_t get_kPreBlockFactor() const { return kPreBlockFactor; }
     uint64_t get_kPreBlockTotal() const { return kPreBlockTotal; }
     uint32_t get_kPreBlockTail() const { return kPreBlockTail; }
     uint32_t get_vPreBlockFactor() const { return vPreBlockFactor; }
     uint64_t get_vPreBlockTotal() const { return vPreBlockTotal; }
     uint32_t get_vPreBlockTail() const { return vPreBlockTail; }
     uint32_t get_maskCoreNum() const { return maskCoreNum; }
     uint32_t get_castBufferLen() const { return castBufferLen; }
     uint32_t get_outputBufferLen() const { return outputBufferLen; }
     uint32_t get_inputBufferLen() const { return inputBufferLen; }
     uint32_t get_singleUBProcessNum() const { return singleUBProcessNum; }
     uint32_t get_maskSingleCoreLoop() const { return maskSingleCoreLoop; }
     uint32_t get_maskLastLoopNum() const { return maskLastLoopNum; }
     uint32_t get_maskTailCoreLoop() const { return maskTailCoreLoop; }
     uint32_t get_maskTailCoreLastLoopNum() const { return maskTailCoreLastLoopNum; }
     uint32_t get_dropoutIsDivisibleBy8() const { return dropoutIsDivisibleBy8; }
     bool get_sValueZeroUnderTND() const { return sValueZeroUnderTND; }
 
     void set_maskPreBlockTotal(uint64_t val) { maskPreBlockTotal = val; }
     void set_maskSingleCoreNum(uint64_t val) { maskSingleCoreNum = val; }
     void set_qPreBlockFactor(uint32_t val) { qPreBlockFactor = val; }
     void set_qPreBlockTotal(uint64_t val) { qPreBlockTotal = val; }
     void set_qPreBlockTail(uint32_t val) { qPreBlockTail = val; }
     void set_kPreBlockFactor(uint32_t val) { kPreBlockFactor = val; }
     void set_kPreBlockTotal(uint64_t val) { kPreBlockTotal = val; }
     void set_kPreBlockTail(uint32_t val) { kPreBlockTail = val; }
     void set_vPreBlockFactor(uint32_t val) { vPreBlockFactor = val; }
     void set_vPreBlockTotal(uint64_t val) { vPreBlockTotal = val; }
     void set_vPreBlockTail(uint32_t val) { vPreBlockTail = val; }
     void set_maskCoreNum(uint32_t val) { maskCoreNum = val; }
     void set_castBufferLen(uint32_t val) { castBufferLen = val; }
     void set_outputBufferLen(uint32_t val) { outputBufferLen = val; }
     void set_inputBufferLen(uint32_t val) { inputBufferLen = val; }
     void set_singleUBProcessNum(uint32_t val) { singleUBProcessNum = val; }
     void set_maskSingleCoreLoop(uint32_t val) { maskSingleCoreLoop = val; }
     void set_maskLastLoopNum(uint32_t val) { maskLastLoopNum = val; }
     void set_maskTailCoreLoop(uint32_t val) { maskTailCoreLoop = val; }
     void set_maskTailCoreLastLoopNum(uint32_t val) { maskTailCoreLastLoopNum = val; }
     void set_dropoutIsDivisibleBy8(uint32_t val) { dropoutIsDivisibleBy8 = val; }
     void set_sValueZeroUnderTND(bool val) { sValueZeroUnderTND = val; }
 };
 
 class PostParamsRegbase {
 public:
     uint32_t postUbBaseSize;
     uint32_t qPostBlockFactor;
     uint64_t qPostBlockTotal;
     uint32_t qPostBaseNum;
     uint32_t qPostTailNum;
     uint32_t kPostBlockFactor;
     uint64_t kPostBlockTotal;
     uint32_t kPostBaseNum;
     uint32_t kPostTailNum;
     uint32_t vPostBlockFactor;
     uint64_t vPostBlockTotal;
     uint32_t vPostBaseNum;
     uint32_t vPostTailNum;
     uint64_t dqWorkSpaceOffset;
     uint64_t dkWorkSpaceOffset;
     uint64_t dvWorkSpaceOffset;
     uint64_t vScaleDsWorkSpaceOffset;
     uint64_t dropMaskGmOffset;
     uint64_t deterGmOffset;
     uint64_t deterWorkSpaceOffset;
 
     uint32_t get_postUbBaseSize() const { return postUbBaseSize; }
     uint32_t get_qPostBlockFactor() const { return qPostBlockFactor; }
     uint64_t get_qPostBlockTotal() const { return qPostBlockTotal; }
     uint32_t get_qPostBaseNum() const { return qPostBaseNum; }
     uint32_t get_qPostTailNum() const { return qPostTailNum; }
     uint32_t get_kPostBlockFactor() const { return kPostBlockFactor; }
     uint64_t get_kPostBlockTotal() const { return kPostBlockTotal; }
     uint32_t get_kPostBaseNum() const { return kPostBaseNum; }
     uint32_t get_kPostTailNum() const { return kPostTailNum; }
     uint32_t get_vPostBlockFactor() const { return vPostBlockFactor; }
     uint64_t get_vPostBlockTotal() const { return vPostBlockTotal; }
     uint32_t get_vPostBaseNum() const { return vPostBaseNum; }
     uint32_t get_vPostTailNum() const { return vPostTailNum; }
     uint64_t get_dqWorkSpaceOffset() const { return dqWorkSpaceOffset; }
     uint64_t get_dkWorkSpaceOffset() const { return dkWorkSpaceOffset; }
     uint64_t get_dvWorkSpaceOffset() const { return dvWorkSpaceOffset; }
     uint64_t get_vScaleDsWorkSpaceOffset() const { return vScaleDsWorkSpaceOffset; }
     uint64_t get_dropMaskGmOffset() const { return dropMaskGmOffset; }
     uint64_t get_deterGmOffset() const { return deterGmOffset; }
     uint64_t get_deterWorkSpaceOffset() const { return deterWorkSpaceOffset; }
 
     void set_postUbBaseSize(uint32_t value) { postUbBaseSize = value; }
     void set_qPostBlockFactor(uint32_t value) { qPostBlockFactor = value; }
     void set_qPostBlockTotal(uint64_t value) { qPostBlockTotal = value; }
     void set_qPostBaseNum(uint32_t value) { qPostBaseNum = value; }
     void set_qPostTailNum(uint32_t value) { qPostTailNum = value; }
     void set_kPostBlockFactor(uint32_t value) { kPostBlockFactor = value; }
     void set_kPostBlockTotal(uint64_t value) { kPostBlockTotal = value; }
     void set_kPostBaseNum(uint32_t value) { kPostBaseNum = value; }
     void set_kPostTailNum(uint32_t value) { kPostTailNum = value; }
     void set_vPostBlockFactor(uint32_t value) { vPostBlockFactor = value; }
     void set_vPostBlockTotal(uint64_t value) { vPostBlockTotal = value; }
     void set_vPostBaseNum(uint32_t value) { vPostBaseNum = value; }
     void set_vPostTailNum(uint32_t value) { vPostTailNum = value; }
     void set_dqWorkSpaceOffset(uint64_t value) { dqWorkSpaceOffset = value; }
     void set_dkWorkSpaceOffset(uint64_t value) { dkWorkSpaceOffset = value; }
     void set_dvWorkSpaceOffset(uint64_t value) { dvWorkSpaceOffset = value; }
     void set_vScaleDsWorkSpaceOffset(uint64_t value) { vScaleDsWorkSpaceOffset = value; }
     void set_dropMaskGmOffset(uint64_t value) { dropMaskGmOffset = value; }
     void set_deterGmOffset(uint64_t value) { deterGmOffset = value; }
     void set_deterWorkSpaceOffset(uint64_t value) { deterWorkSpaceOffset = value; }
 };

 class DeterParamRegbase {
 public:
     constexpr static int64_t DETER_PREFIX_NUM = 132;

     bool coreDivide;
     int64_t deterPrefixStep;
     int64_t deterPrefix[DETER_PREFIX_NUM];
     int64_t deterPrefixAlign[DETER_PREFIX_NUM];
     int64_t deterPrefix0[DETER_PREFIX_NUM];
     int64_t deterPrefix1[DETER_PREFIX_NUM];
     int64_t deterPrefix2[DETER_PREFIX_NUM];

     bool get_coreDivide() const { return coreDivide; }
     int64_t get_deterPrefixStep() const { return deterPrefixStep; }
     const int64_t *get_deterPrefix() const { return deterPrefix; }
     int64_t get_deterPrefix(int index) const { return deterPrefix[index]; }
     const int64_t *get_deterPrefixAlign() const { return deterPrefixAlign; }
     int64_t get_deterPrefixAlign(int index) const { return deterPrefixAlign[index]; }
     const int64_t *get_deterPrefix0() const { return deterPrefix0; }
     int64_t get_deterPrefix0(int index) const { return deterPrefix0[index]; }
     const int64_t *get_deterPrefix1() const { return deterPrefix1; }
     int64_t get_deterPrefix1(int index) const { return deterPrefix1[index]; }
     const int64_t *get_deterPrefix2() const { return deterPrefix2; }
     int64_t get_deterPrefix2(int index) const { return deterPrefix2[index]; }
     void set_coreDivide(bool value) { coreDivide = value; }
     void set_deterPrefixStep(int64_t value) { deterPrefixStep = value; }
     void set_deterPrefix(const int64_t *val) {
         for (int i = 0; i < DETER_PREFIX_NUM; ++i) {
             deterPrefix[i] = val[i];
         }
     }
     void set_deterPrefix(int index, int64_t val) { deterPrefix[index] = val; }
     void set_deterPrefixAlign(const int64_t *val) {
         for (int i = 0; i < DETER_PREFIX_NUM; ++i) {
             deterPrefixAlign[i] = val[i];
         }
     }
     void set_deterPrefixAlign(int index, int64_t val) { deterPrefixAlign[index] = val; }
     void set_deterPrefix0(const int64_t *val) {
         for (int i = 0; i < DETER_PREFIX_NUM; ++i) {
             deterPrefix0[i] = val[i];
         }
     }
     void set_deterPrefix0(int index, int64_t val) { deterPrefix0[index] = val; }
     void set_deterPrefix1(const int64_t *val) {
         for (int i = 0; i < DETER_PREFIX_NUM; ++i) {
             deterPrefix1[i] = val[i];
         }
     }
     void set_deterPrefix1(int index, int64_t val) { deterPrefix1[index] = val; }
     void set_deterPrefix2(const int64_t *val) {
         for (int i = 0; i < DETER_PREFIX_NUM; ++i) {
             deterPrefix2[i] = val[i];
         }
     }
     void set_deterPrefix2(int index, int64_t val) { deterPrefix2[index] = val; }
 };

 class TndParamRegbase {
 public:
    uint64_t tndStartBIdx[MAX_CORE_NUM];
    uint64_t tndS1S2PrefixSum[MAX_CORE_NUM];
    uint64_t tndS1S2AlignPrefixSum[MAX_CORE_NUM];
    uint64_t tndPrefixSum[MAX_CORE_NUM];
    void set_tndStartBIdx(const uint64_t *val)
     {
         for (int i = 0; i < MAX_CORE_NUM; ++i) {
             tndStartBIdx[i] = val[i];
         }
     }
     void set_tndS1S2PrefixSum(const uint64_t *val)
     {
         for (int i = 0; i < MAX_CORE_NUM; ++i) {
             tndS1S2PrefixSum[i] = val[i];
         }
     }
     void set_tndS1S2AlignPrefixSum(const uint64_t *val)
     {
         for (int i = 0; i < MAX_CORE_NUM; ++i) {
             tndS1S2AlignPrefixSum[i] = val[i];
         }
     }
     void set_tndPrefixSum(const uint64_t *val)
     {
         for (int i = 0; i < MAX_CORE_NUM; ++i) {
             tndPrefixSum[i] = val[i];
         }
     }
 };

 template<const bool isNewDeter = false, const bool isTnd = false>
 class FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase {
 public:
     FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase s1s2BNGS1S2BaseParams;
     FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParamsRegbase s1s2BNGS1S2SplitCoreParams;
     BlockNumListParamsRegbase s1s2BNGS1S2BlockNumList;
     PreParamsRegbase preTilingData;
     PostParamsRegbase postTilingData;
     typename std::conditional<isNewDeter, DeterParamRegbase, std::nullptr_t>::type deterParam;
     typename std::conditional<!isNewDeter && isTnd, TndParamRegbase, std::nullptr_t>::type tndParam;
 };
 }  // namespace fag
 }  // namespace optiling
 #endif
