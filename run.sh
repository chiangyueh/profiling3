#!/bin/bash
# NEW BEGIN
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

if [[ ! "${FA_NPU_ID:-}" =~ ^[0-9]+$ ]]; then
    echo "invalid FA_NPU_ID: ${FA_NPU_ID:-}" >&2
    exit 2
fi
if [[ ! "${FA_TILING_KEY:-}" =~ ^[0-9]+$ ]]; then
    echo "invalid FA_TILING_KEY: ${FA_TILING_KEY:-}" >&2
    exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES="${FA_NPU_ID}"
ASCEND_ROOT=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
if [[ -f "${ASCEND_ROOT}/bin/setenv.bash" ]]; then
    source "${ASCEND_ROOT}/bin/setenv.bash"
fi

EXPECTED_COMMIT=6ead121aded45355043b502756b6592fd7c30b14
SOC_UNIT=${FA_SOC_UNIT:-ascend910b}
CACHE_BASE=${FA_CACHE_ROOT:-"${SCRIPT_DIR}/out/fa_cache"}
CACHE_DIR="${CACHE_BASE}/${EXPECTED_COMMIT}/${SOC_UNIT}/${FA_TILING_KEY}"
INSTALL_ROOT="${CACHE_DIR}/opp"
BINARY="${CACHE_DIR}/fa_npu"

if [[ "${FA_SKIP_BUILD:-0}" != "1" ]]; then
    bash "${SCRIPT_DIR}/build.sh"
fi

SET_ENV=$(find "${INSTALL_ROOT}/vendors" -mindepth 3 -maxdepth 3 -type f \
    -path '*/bin/set_env.bash' -print 2>/dev/null | head -n 1 || true)
if [[ -z "${SET_ENV}" || ! -x "${BINARY}" ]]; then
    echo "FA build cache is incomplete for tiling key ${FA_TILING_KEY}" >&2
    exit 2
fi

source "${SET_ENV}"
export FA_CUSTOM_OPP_ROOT="${SET_ENV%/bin/set_env.bash}"

if [[ "${FA_PROFILE:-1}" == "0" ]]; then
    exec "${BINARY}"
fi
exec msprof op "${BINARY}"
# NEW END
