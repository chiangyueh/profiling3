#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Exact shape and key from the official v8.5.0 FAG V2 example/host tiling test.
TAG=v8.5.0
COMMIT=6ead121aded45355043b502756b6592fd7c30b14
TILING_KEY=134258996
SHAPE="B=1,N1=1,N2=1,S1=256,S2=256,D=128,dtype=FP32,layout=SBH"
SOC=${FAG_SOC:-ascend910b}
VENDOR=fag_single_key
JOBS=${FAG_BUILD_JOBS:-1}
RUN_TIMEOUT=${FAG_RUN_TIMEOUT:-300}

SNAPSHOT="${SCRIPT_DIR}/vendor/ops-transformer-v8.5.0/attention/flash_attention_score_grad"
DEFAULT_OPS_ROOT="${SCRIPT_DIR}/../ops-transformer-official-8.5.0"
OPS_ROOT=${OPS_TRANSFORMER_ROOT:-${DEFAULT_OPS_ROOT}}
BUILD_ROOT=${FAG_BUILD_ROOT:-"${SCRIPT_DIR}/out/fag_npu"}
RESULT_ROOT=${FAG_RESULT_ROOT:-"${SCRIPT_DIR}/results/fag_npu"}
INSTALL_ROOT="${BUILD_ROOT}/opp"

if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] FAG_BUILD_JOBS must be a positive integer" >&2
    exit 2
fi
if [[ ! "${RUN_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] FAG_RUN_TIMEOUT must be a positive integer (seconds)" >&2
    exit 2
fi

mkdir -p "${RESULT_ROOT}" "${BUILD_ROOT}" "${INSTALL_ROOT}"
exec > >(tee "${RESULT_ROOT}/full.log") 2>&1

echo "[INFO] source: ops-transformer ${TAG} (${COMMIT})"
echo "[INFO] operator: FlashAttentionScoreGrad"
echo "[INFO] one shape: ${SHAPE}"
echo "[INFO] one tiling key: ${TILING_KEY}"
echo "[INFO] build jobs: ${JOBS}"

if ! git -C "${OPS_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    OPS_ROOT="${BUILD_ROOT}/ops-transformer-${TAG}"
    if [[ -e "${OPS_ROOT}" ]] && ! git -C "${OPS_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "[ERROR] ${OPS_ROOT} exists but is not a Git repository" >&2
        exit 2
    fi
    if [[ ! -d "${OPS_ROOT}/.git" ]]; then
        echo "[INFO] downloading the official ${TAG} source"
        git clone --depth 1 --branch "${TAG}" \
            https://gitcode.com/cann/ops-transformer.git "${OPS_ROOT}"
    fi
fi

ACTUAL_COMMIT=$(git -C "${OPS_ROOT}" rev-parse HEAD)
if [[ "${ACTUAL_COMMIT}" != "${COMMIT}" ]]; then
    echo "[ERROR] expected ops-transformer ${COMMIT}, got ${ACTUAL_COMMIT}" >&2
    exit 2
fi

if ! git -C "${OPS_ROOT}" diff --quiet -- attention/flash_attention_score_grad || \
   ! git -C "${OPS_ROOT}" diff --cached --quiet -- attention/flash_attention_score_grad; then
    echo "[ERROR] the official FAG source tree has tracked modifications" >&2
    exit 2
fi

if ! diff -qr "${SNAPSHOT}" \
    "${OPS_ROOT}/attention/flash_attention_score_grad" >/dev/null; then
    echo "[ERROR] the build source does not match the vendored official FAG snapshot" >&2
    exit 2
fi

ASCEND_ROOT=${ASCEND_HOME_PATH:-}
if [[ -z "${ASCEND_ROOT}" ]]; then
    for candidate in \
        /usr/local/Ascend/cann-8.5.0 \
        /usr/local/Ascend/ascend-toolkit/latest \
        /usr/local/Ascend/latest; do
        if [[ -f "${candidate}/bin/setenv.bash" || -f "${candidate}/set_env.sh" ]]; then
            ASCEND_ROOT=${candidate}
            break
        fi
    done
fi

if [[ -f "${ASCEND_ROOT}/bin/setenv.bash" ]]; then
    CANN_SETENV="${ASCEND_ROOT}/bin/setenv.bash"
elif [[ -f "${ASCEND_ROOT}/set_env.sh" ]]; then
    CANN_SETENV="${ASCEND_ROOT}/set_env.sh"
else
    echo "[ERROR] CANN environment script was not found; set ASCEND_HOME_PATH" >&2
    exit 2
fi

export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "${CANN_SETENV}"
ASCEND_ROOT=${ASCEND_HOME_PATH:-${ASCEND_ROOT}}

CMAKE_MIN=3.21
if command -v cmake >/dev/null 2>&1; then
    CMAKE_VERSION=$(cmake --version | awk 'NR == 1 {print $3}')
else
    CMAKE_VERSION=
fi
if [[ -z "${CMAKE_VERSION}" ]] || \
   [[ "$(printf '%s\n' "${CMAKE_MIN}" "${CMAKE_VERSION}" | sort -V | sed -n '1p')" != "${CMAKE_MIN}" ]]; then
    CMAKE_ROOT="${BUILD_ROOT}/cmake-3.28.3"
    CMAKE_BIN="${CMAKE_ROOT}/cmake/data/bin"
    if [[ ! -x "${CMAKE_BIN}/cmake" ]]; then
        echo "[INFO] installing CMake 3.28.3 locally under ${CMAKE_ROOT}"
        python3 -m pip install --no-cache-dir --target "${CMAKE_ROOT}" cmake==3.28.3
    fi
    export PATH="${CMAKE_BIN}:${PATH}"
fi

echo "[INFO] using $(cmake --version | sed -n '1p')"
export CMAKE_BUILD_PARALLEL_LEVEL=${JOBS}
export MAKEFLAGS="-j${JOBS}"
export MAX_JOBS=${JOBS}

BUILD_PREFIX=()
if command -v ionice >/dev/null 2>&1; then
    BUILD_PREFIX+=(ionice -c 3)
fi
if command -v nice >/dev/null 2>&1; then
    BUILD_PREFIX+=(nice -n 15)
fi

echo "[INFO] compiling only tiling key ${TILING_KEY} (no JIT, no all-key build)"
(
    cd "${OPS_ROOT}"
    "${BUILD_PREFIX[@]}" bash build.sh \
        --pkg \
        --ops=flash_attention_score_grad \
        --soc="${SOC}" \
        --vendor_name="${VENDOR}" \
        --tiling_key="${TILING_KEY}" \
        -j"${JOBS}"
)

PACKAGE=$(find "${OPS_ROOT}/output" "${OPS_ROOT}/build" \
    -type f -name '*.run' -printf '%T@ %p\n' 2>/dev/null | \
    sort -nr | sed -n '1p' | cut -d' ' -f2-)
if [[ -z "${PACKAGE}" ]]; then
    echo "[ERROR] the official build produced no .run package" >&2
    exit 1
fi

echo "[INFO] installing the one-key package into ${INSTALL_ROOT}"
"${PACKAGE}" --quiet --install-path="${INSTALL_ROOT}"

CUSTOM_ROOT="${INSTALL_ROOT}/vendors/${VENDOR}_transformer"
if [[ ! -f "${CUSTOM_ROOT}/bin/set_env.bash" ]]; then
    echo "[ERROR] installed custom FAG tree was not found at ${CUSTOM_ROOT}" >&2
    exit 1
fi
source "${CUSTOM_ROOT}/bin/set_env.bash"

ARCH=$(uname -m)
SAMPLE="${OPS_ROOT}/attention/flash_attention_score_grad/examples/test_aclnn_flash_attention_score_grad_v2.cpp"
BIN="${BUILD_ROOT}/fag_official_v2"

echo "[INFO] compiling the official one-shape launcher"
g++ -O2 -std=c++17 "${SAMPLE}" \
    -I"${ASCEND_ROOT}/include" \
    -I"${CUSTOM_ROOT}/op_api/include" \
    -I"${ASCEND_ROOT}/${ARCH}-linux/include/aclnnop" \
    -L"${CUSTOM_ROOT}/op_api/lib" \
    -L"${ASCEND_ROOT}/lib64" \
    -Wl,-rpath,"${CUSTOM_ROOT}/op_api/lib" \
    -Wl,-rpath,"${ASCEND_ROOT}/lib64" \
    -lcust_opapi -lascendcl -lnnopbase -lpthread -lc_sec \
    -o "${BIN}"

echo "[INFO] running exactly one FAG shape on NPU (timeout: ${RUN_TIMEOUT}s)"
set +e
timeout --signal=INT --kill-after=10 "${RUN_TIMEOUT}" "${BIN}" \
    2>&1 | tee "${RESULT_ROOT}/run.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

if [[ ${RUN_STATUS} -eq 0 ]]; then
    echo "PASS" > "${RESULT_ROOT}/status.txt"
    echo "[PASS] one official FAG shape completed"
    echo "[PASS] output: ${RESULT_ROOT}/run.log"
elif [[ ${RUN_STATUS} -eq 124 ]]; then
    echo "TIMEOUT (${RUN_TIMEOUT}s)" > "${RESULT_ROOT}/status.txt"
    echo "[FAIL] the one-shape NPU run exceeded ${RUN_TIMEOUT}s" >&2
else
    echo "FAIL (${RUN_STATUS})" > "${RESULT_ROOT}/status.txt"
    echo "[FAIL] the one-shape NPU run exited with ${RUN_STATUS}" >&2
fi

exit "${RUN_STATUS}"
