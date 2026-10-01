#!/bin/bash
set -euo pipefail

CURRENT_DIR=$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)
cd "${CURRENT_DIR}"

RUN_MODE="sim"
CYCLES_ONLY=0
NPU_ID=""
SHORT=r:,v:
LONG=run-mode:,soc-version:,cycles-only,id:
OPTS=$(getopt -a --options "${SHORT}" --longoptions "${LONG}" -- "$@")
eval set -- "${OPTS}"

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
    --id)
        NPU_ID="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR]: Unexpected option: $1" >&2
        exit 1
        ;;
    esac
done

if [[ -n "${NPU_ID}" ]]; then
    if [[ ! "${NPU_ID}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR]: --id must be a non-negative integer" >&2
        exit 2
    fi
    export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID}"
elif [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    echo "[ERROR]: NPU is not selected; pass --id=<NPU_ID>" >&2
    exit 2
fi

if [[ "${RUN_MODE}" != "npu" ]]; then
    echo "[ERROR]: FA/FAG runner currently supports -r npu only" >&2
    exit 2
fi

echo "[INFO] physical NPU: ${ASCEND_RT_VISIBLE_DEVICES} (launcher logical device: 0)"

ASCEND_ROOT=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
if [[ -f "${ASCEND_ROOT}/bin/setenv.bash" ]]; then
    source "${ASCEND_ROOT}/bin/setenv.bash"
fi

EXPECTED_COMMIT=6ead121aded45355043b502756b6592fd7c30b14
SOC_UNIT=${ATTENTION_SOC_UNIT:-ascend910b}
TILING_KEY=${ATTENTION_TILING_KEY:-74804}
OPERATOR=${ATTENTION_OPERATOR:-flash_attention_score_grad}
CACHE_BASE=${ATTENTION_KERNEL_CACHE_ROOT:-"${CURRENT_DIR}/out/attention_cache"}
CACHE_DIR="${CACHE_BASE}/${EXPECTED_COMMIT}/${SOC_UNIT}/${OPERATOR}/${TILING_KEY}"
INSTALL_ROOT="${CACHE_DIR}/opp"
ATTENTION_BINARY="${CACHE_DIR}/attention_npu"

bash "${CURRENT_DIR}/build_attention.sh"

SET_ENV=""
if [[ -d "${INSTALL_ROOT}/vendors" ]]; then
    SET_ENV=$(find "${INSTALL_ROOT}/vendors" -mindepth 3 -maxdepth 3 -type f \
        -path '*/bin/set_env.bash' -print | head -n 1)
fi
if [[ -n "${SET_ENV}" ]]; then
    source "${SET_ENV}"
fi

if [[ ! -x "${ATTENTION_BINARY}" ]]; then
    echo "[ERROR]: cached attention evaluator is missing: ${ATTENTION_BINARY}" >&2
    exit 2
fi

if [[ "${CYCLES_ONLY}" == "1" ]]; then
    echo "[ERROR]: --cycles-only is not implemented for the ACLNN attention runner" >&2
    exit 2
fi

if [[ "${ATTENTION_REFERENCE:-0}" == "1" ]]; then
    exec "${ATTENTION_BINARY}"
fi

exec msprof op "${ATTENTION_BINARY}"
