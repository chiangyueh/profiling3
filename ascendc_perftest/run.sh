#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))

cd ${SCRIPT_DIR}

export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH

EXEC_BIN=${PROJECT_ROOT}/build/bin/ascendc_perftest

# 默认测试类型为put
TEST_TYPE="put"
# 默认数据类型为float
DATA_TYPE="float"
# 默认核数范围
MIN_BLOCK_SIZE="32"
MAX_BLOCK_SIZE="32"
BLOCK_LIST=""
# 默认幂数范围
MIN_EXPONENT="3"
MAX_EXPONENT="20"
# 默认设备ID
DEVICE1="1"
DEVICE2="2"
# 默认循环次数
LOOP_COUNT="1000"
# 默认UB size(KB)
UB_SIZE="16"
# 分析模式: none/plot/md (none表示都不生成, plot表示只生成图, md表示同时生成图和md), 默认none
ANALYSE_MODE="none"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test-type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -d|--datatype)
            DATA_TYPE="$2"
            shift 2
            ;;
        --device1)
            DEVICE1="$2"
            shift 2
            ;;
        --device2)
            DEVICE2="$2"
            shift 2
            ;;
        --loop-count)
            LOOP_COUNT="$2"
            shift 2
            ;;
        --ub-size)
            UB_SIZE="$2"
            shift 2
            ;;
        -b|--block-size)
            MIN_BLOCK_SIZE="$2"
            MAX_BLOCK_SIZE="$2"
            BLOCK_LIST=""
            shift 2
            ;;
        --block-range)
            MIN_BLOCK_SIZE="$2"
            MAX_BLOCK_SIZE="$3"
            BLOCK_LIST=""
            shift 3
            ;;
        --block-list)
            if [ -n "$2" ]; then
                BLOCK_LIST="$2"
                shift 2
            else
                echo "Error: --block-list requires a value."
                exit 1
            fi
            ;;
        -e|--exponent)
            MIN_EXPONENT="$2"
            MAX_EXPONENT="$2"
            shift 2
            ;;
        --exponent-range)
            MIN_EXPONENT="$2"
            MAX_EXPONENT="$3"
            shift 3
            ;;
        -a|--analyse)
            if [ -n "$2" ]; then
                ANALYSE_MODE="$2"
                shift 2
            else
                echo "Error: -a|--analyse requires a value."
                exit 1
            fi
            ;;
        *)
            echo "未知参数: $1"
            echo "使用方法: $0 [-t|--test-type] <put|get|ub2gm_local|ub2gm_remote|gm2ub_local|gm2ub_remote|all> [-d|--datatype] <float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char|all> [选项]"
            echo "选项:"
            echo "  -t|--test-type <type>           设置测试类型 (put|get|ub2gm_local|ub2gm_remote|gm2ub_local|gm2ub_remote|all)"
            echo "  -d|--datatype <type>            设置数据类型 (float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char|all)"
            echo "  -b|--block-size <size>          设置核数"
            echo "  --block-range <min> <max>       设置连续核数范围"
            echo "  --block-list <b1,b2,...>        设置离散核数列表，如 2,4,6,8"
            echo "  -e|--exponent <exponent>        设置数据量的幂数"
            echo "  --exponent-range <min> <max>    设置数据量的幂数范围"
            echo "  --device1 <id>                  设置设备1 ID"
            echo "  --device2 <id>                  设置设备2 ID"
            echo "  --loop-count <count>            设置循环次数"
            echo "  --ub-size <size>                设置UB size(KB), 默认16"
            echo "  -a|--analyse <none|plot|md>    设置分析模式 (none=不生成, plot=只生成图, md=同时生成图和md), 默认none"
            exit 1
            ;;
    esac
done

# 验证测试类型
VALID_TEST_TYPES="put get ub2gm_local ub2gm_remote gm2ub_local gm2ub_remote all"
if [[ ! " $VALID_TEST_TYPES " =~ " $TEST_TYPE " ]]; then
    echo "错误: 测试类型必须是 'put'、'get'、'ub2gm_local'、'ub2gm_remote'、'gm2ub_local'、'gm2ub_remote' 或 'all'"
    echo "使用方法: $0 [-t|--test-type] <put|get|ub2gm_local|ub2gm_remote|gm2ub_local|gm2ub_remote|all> [-d|--datatype] <float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char|all> [选项]"
    exit 1
fi

# 验证数据类型
VALID_DATATYPES="float int8 int16 int32 int64 uint8 uint16 uint32 uint64 char all"
if [[ ! " $VALID_DATATYPES " =~ " $DATA_TYPE " ]]; then
    echo "错误: 数据类型必须是 'float'、'int8'、'int16'、'int32'、'int64'、'uint8'、'uint16'、'uint32'、'uint64'、'char' 或 'all'"
    echo "使用方法: $0 [-t|--test-type] <put|get|ub2gm_local|ub2gm_remote|gm2ub_local|gm2ub_remote|all> [-d|--datatype] <float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char|all> [选项]"
    exit 1
fi

echo "测试类型: $TEST_TYPE"
echo "数据类型: $DATA_TYPE"
if [ -n "$BLOCK_LIST" ]; then
    echo "核数列表: $BLOCK_LIST"
else
    echo "核数范围: $MIN_BLOCK_SIZE-$MAX_BLOCK_SIZE"
fi
echo "幂数范围: $MIN_EXPONENT-$MAX_EXPONENT"
echo "UB size(KB): $UB_SIZE"

# 定义所有数据类型数组
ALL_DATATYPES=("float" "int8" "int16" "int32" "int64" "uint8" "uint16" "uint32" "uint64" "char")

ALL_TEST_TYPES=("put" "get" "ub2gm_local" "ub2gm_remote" "gm2ub_local" "gm2ub_remote")

run_test() {
    local test_type="$1"
    local data_type="$2"
    local block_args=()
    if [ -n "$BLOCK_LIST" ]; then
        block_args=(--block-list "$BLOCK_LIST")
    else
        block_args=(--block-range "$MIN_BLOCK_SIZE" "$MAX_BLOCK_SIZE")
    fi
    ${EXEC_BIN} -t "$test_type" -d "$data_type" "${block_args[@]}" --exponent-range "$MIN_EXPONENT" "$MAX_EXPONENT" --device1 $DEVICE1 --device2 $DEVICE2 --loop-count $LOOP_COUNT --ub-size $UB_SIZE
}

# 执行测试
if [[ "$TEST_TYPE" == "all" && "$DATA_TYPE" == "all" ]]; then
    echo "正在运行所有测试类型和数据类型..."
    for type in "${ALL_TEST_TYPES[@]}"; do
        for dtype in "${ALL_DATATYPES[@]}"; do
            echo "
=== 运行测试类型: $type, 数据类型: $dtype ==="
            run_test "$type" "$dtype"
        done
    done
elif [[ "$TEST_TYPE" == "all" ]]; then
    echo "正在运行所有测试类型，数据类型: $DATA_TYPE..."
    for type in "${ALL_TEST_TYPES[@]}"; do
        echo "
=== 运行测试类型: $type, 数据类型: $DATA_TYPE ==="
        run_test "$type" "$DATA_TYPE"
    done
elif [[ "$DATA_TYPE" == "all" ]]; then
    echo "正在运行测试类型: $TEST_TYPE，所有数据类型..."
    for dtype in "${ALL_DATATYPES[@]}"; do
        echo "
=== 运行测试类型: $TEST_TYPE, 数据类型: $dtype ==="
        run_test "$TEST_TYPE" "$dtype"
    done
else
    run_test "$TEST_TYPE" "$DATA_TYPE"
fi

cd ${CURRENT_DIR}

# 运行 perf_data_process.py 处理 output 目录
PERF_SCRIPT="${SCRIPT_DIR}/../../utils/perf_data_process.py"
if [ -f "${PERF_SCRIPT}" ]; then
    if [ "$ANALYSE_MODE" = "plot" ] || [ "$ANALYSE_MODE" = "md" ]; then
        echo -e "\n========== Generating performance charts =========="
        cmd_args=("-d" "${SCRIPT_DIR}/output")
        if [ "$ANALYSE_MODE" = "md" ]; then
            # md模式，图和md都生成
            true
        else
            # plot模式，只生成图，不生成md
            cmd_args+=("--no-markdown")
        fi
        python3 "${PERF_SCRIPT}" "${cmd_args[@]}"
    fi
fi