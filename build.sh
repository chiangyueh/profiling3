#!/bin/bash
# Build and run CCE benchmarks (msprof op simulator).
# Артефакты изолированы по RUN_MODE -> можно держать cpu/npu/sim одновременно.
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd "$CURRENT_DIR"

BUILD_TYPE="Debug"
SOC_VERSION="Ascend310P3"
RUN_MODE="sim"
KERNEL_TYPE="all"  # Значение по умолчанию
INSTALL_PREFIX=""   # по умолчанию вычислим из RUN_MODE ниже

# Исправлен SHORT: добавлен t: и убрана лишняя запятая
SHORT=r:,v:,i:,b:,p:,t:
LONG=run-mode:,soc-version:,install-path:,build-type:,install-prefix:,kernel-type:
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
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
        ;;
    -b | --build-type)
        BUILD_TYPE="$2"
        shift 2
        ;;
    -p | --install-prefix)
        INSTALL_PREFIX="$2"
        shift 2
        ;;
    -t | --kernel-type)
        KERNEL_TYPE="$2"
        shift 2
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

RUN_MODE_LIST="sim npu cpu"
if [[ " $RUN_MODE_LIST " != *" $RUN_MODE "* ]]; then
    echo "[ERROR]: benchmarks support sim or npu. Example: bash start.sh -r sim -v Ascend310P3"
    exit 1
fi

VERSION_LIST="Ascend910A Ascend910B Ascend310B1 Ascend310B2 Ascend310B3 Ascend310B4 Ascend310P1 Ascend310P3 Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4"
if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "[ERROR]: SOC_VERSION should be in [$VERSION_LIST]"
    exit 1
fi


KERNEL_TYPE_LIST="base singlecoreSplitK deterministicSplitK all"
if [[ " $KERNEL_TYPE_LIST " != *" $KERNEL_TYPE "* ]]; then
    echo "[ERROR]: KERNEL_TYPE should be in [$KERNEL_TYPE_LIST]. Example: -t base OR -t splitK OR -t all"
    exit 1
fi

# --- Каталоги/имена, зависящие от RUN_MODE (изоляция параллельных сборок) ---
BUILD_DIR="build_${RUN_MODE}"
# install-префикс: если не задан явно через -p, вычисляем как out/<mode>
if [ -z "$INSTALL_PREFIX" ]; then
    INSTALL_PREFIX="${CURRENT_DIR}/out/${RUN_MODE}"
fi
BIN_NAME="matmul_${RUN_MODE}"

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
echo ASCEND=${_ASCEND_INSTALL_PATH}
export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
echo "[INFO]: benchmarks — RUN_MODE=${RUN_MODE} SOC=${SOC_VERSION} KERNEL=${KERNEL_TYPE}"
echo "[INFO]: build=${BUILD_DIR} install=${INSTALL_PREFIX} bin=${BIN_NAME}"
source "${_ASCEND_INSTALL_PATH}/bin/setenv.bash"

if [ "${RUN_MODE}" = "sim" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

set -e
# чистим ТОЛЬКО каталоги текущего режима (не трогаем другие сборки)
rm -rf "${BUILD_DIR}" "${INSTALL_PREFIX}"
mkdir -p output
cmake -B "${BUILD_DIR}" \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DKERNEL_TYPE=${KERNEL_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
cmake --build "${BUILD_DIR}" -j
cmake --install "${BUILD_DIR}"

# копируем бинарник под именем с режимом (не затирает бинарники других режимов)
rm -f "${BIN_NAME}"
cp "${INSTALL_PREFIX}/bin/${BIN_NAME}" "./${BIN_NAME}"

export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:${INSTALL_PREFIX}/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH

echo "[INFO]: собрано -> ./${BIN_NAME} (install: ${INSTALL_PREFIX})"
