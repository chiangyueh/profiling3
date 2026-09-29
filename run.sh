#!/bin/bash
export MM_KERNEL_TYPE=${MM_KERNEL_TYPE:-2}
export MM_M=${MM_M:-64}
export MM_N=${MM_N:-7168}
export MM_K=${MM_K:-65536}

export MM_CORE_NUM=${MM_CORE_NUM:-20}
export MM_SINGLE_M=${MM_SINGLE_M:-384}
export MM_SINGLE_N=${MM_SINGLE_N:-3584}
export MM_SINGLE_K=${MM_SINGLE_K:-384}
export MM_BASE_M=${MM_BASE_M:-128}
export MM_BASE_N=${MM_BASE_N:-128}
export MM_BASE_K=${MM_BASE_K:-128}
export MM_STEP_M=${MM_STEP_M:-3}
export MM_STEP_N=${MM_STEP_N:-1}
export MM_STEP_Ka=${MM_STEP_Ka:-3}
export MM_STEP_Kb=${MM_STEP_Kb:-3}

export MM_DEPTH_A1=${MM_DEPTH_A1:-9}
export MM_DEPTH_B1=${MM_DEPTH_B1:-6}
export MM_M_TILE_BLOCK=${MM_M_TILE_BLOCK:-1}
export MM_N_TILE_BLOCK=${MM_N_TILE_BLOCK:-1}
export MM_DB_L0A=${MM_DB_L0A:-2}
export MM_DB_L0B=${MM_DB_L0B:-2}
export MM_DB_L0C=${MM_DB_L0C:-2}
export MM_ITERATE_ORDER=${MM_ITERATE_ORDER:-1}
export MM_CAL_ORDER=${MM_CAL_ORDER:-0}

export MM_ND2NZ_A=${MM_ND2NZ_A:-1}
export MM_ND2NZ_B=${MM_ND2NZ_B:-0}
export MM_ND2NZ_BASE_AN=${MM_ND2NZ_BASE_AN:-23}
export MM_ND2NZ_BASE_AD=${MM_ND2NZ_BASE_AD:-2048}
export MM_ND2NZ_BASE_BN=${MM_ND2NZ_BASE_BN:-0}
export MM_ND2NZ_BASE_BD=${MM_ND2NZ_BASE_BD:-0}

export MM_IS_BIAS=${MM_IS_BIAS:-0}
export MM_TRANS_LENGTH=${MM_TRANS_LENGTH:-0}
export MM_TRANS_A=${MM_TRANS_A:-0}
export MM_TRANS_B=${MM_TRANS_B:-0}
export MM_IS_NZ_A=${MM_IS_NZ_A:-0}
export MM_IS_NZ_B=${MM_IS_NZ_B:-0}
export MM_IS_HF32=${MM_IS_HF32:-0}

export MM_PRINT_PLATFORM=${MM_PRINT_PLATFORM:-0}

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd "$CURRENT_DIR"

SOC_VERSION="Ascend910B3"
RUN_MODE="sim"
CYCLES_ONLY=0
SHORT=r:,v:
LONG=run-mode:,soc-version:,cycles-only
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -r | --run-mode)
        RUN_MODE="$2"
        shift 2
        ;;
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    --cycles-only)
        CYCLES_ONLY=1
        shift
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR]: Unexpected option: $1"
        exit 1
        ;;
    esac
done

_ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

source "${_ASCEND_INSTALL_PATH}/bin/setenv.bash"

INSTALL_PREFIX="$(pwd)/out/${RUN_MODE}"
BIN_NAME="matmul_${RUN_MODE}"

export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:${INSTALL_PREFIX}/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH

if [ "${RUN_MODE}" = "sim" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
elif [ "${RUN_MODE}" = "cpu" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${SOC_VERSION}:${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

if [ "${RUN_MODE}" = "npu" ]; then
    if [ "${CYCLES_ONLY}" = 1 ]; then
        # замер системным счётчиком внутри ядра, без профилировщика
        export MM_CYCLES=1
        ./${BIN_NAME}
    else
        msprof op ./${BIN_NAME}
    fi
elif [ "${RUN_MODE}" = "sim" ]; then
    msprof op simulator --soc-version=${SOC_VERSION} ./${BIN_NAME}
elif [ "${RUN_MODE}" = "cpu" ]; then
    ./${BIN_NAME}
fi