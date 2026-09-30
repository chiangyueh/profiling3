#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TAG=v8.5.0
COMMIT=6ead121aded45355043b502756b6592fd7c30b14
SOC=${FAG_BASELINE_SOC:-ascend910b}
VENDOR=fag_baseline
SNAPSHOT="${SCRIPT_DIR}/vendor/ops-transformer-v8.5.0/attention/flash_attention_score_grad"
DEFAULT_OPS_ROOT="${SCRIPT_DIR}/../ops-transformer-official-8.5.0"
OPS_ROOT=${OPS_TRANSFORMER_ROOT:-${DEFAULT_OPS_ROOT}}
RESULT_ROOT=${FAG_BASELINE_RESULT_ROOT:-"${SCRIPT_DIR}/results/fag_baseline"}
BUILD_ROOT=${FAG_BASELINE_BUILD_ROOT:-"${SCRIPT_DIR}/out/fag_baseline"}
INSTALL_ROOT="${BUILD_ROOT}/opp"

mkdir -p "${RESULT_ROOT}" "${BUILD_ROOT}" "${INSTALL_ROOT}"
exec > >(tee "${RESULT_ROOT}/full.log") 2>&1

echo "[INFO] official FAG baseline: ${TAG} (${COMMIT})"

if [[ ! -d "${OPS_ROOT}/.git" ]]; then
    OPS_ROOT="${BUILD_ROOT}/ops-transformer-v8.5.0"
    if [[ -e "${OPS_ROOT}" ]]; then
        echo "[ERROR] ${OPS_ROOT} exists but is not a Git repository" >&2
        exit 2
    fi
    echo "[INFO] cloning the official ops-transformer ${TAG} source"
    git clone --depth 1 --branch "${TAG}" \
        https://gitcode.com/cann/ops-transformer.git "${OPS_ROOT}"
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
    diff -qr "${SNAPSHOT}" "${OPS_ROOT}/attention/flash_attention_score_grad" || true
    exit 2
fi

ASCEND_ROOT=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
if [[ ! -f "${ASCEND_ROOT}/bin/setenv.bash" ]]; then
    echo "[ERROR] CANN setenv.bash not found under ${ASCEND_ROOT}" >&2
    exit 2
fi

export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "${ASCEND_ROOT}/bin/setenv.bash"

echo "[INFO] building only flash_attention_score_grad with one job"
(
    cd "${OPS_ROOT}"
    bash build.sh --pkg \
        --ops=flash_attention_score_grad \
        --soc="${SOC}" \
        --vendor_name="${VENDOR}" \
        -j1
)

PACKAGE_ROOTS=()
for dir in "${OPS_ROOT}/build_out" "${OPS_ROOT}/output" "${OPS_ROOT}/build"; do
    [[ -d "${dir}" ]] && PACKAGE_ROOTS+=("${dir}")
done
if [[ ${#PACKAGE_ROOTS[@]} -eq 0 ]]; then
    echo "[ERROR] the official build produced no package directory" >&2
    exit 1
fi

PACKAGE=$(find "${PACKAGE_ROOTS[@]}" -type f -name '*.run' -printf '%T@ %p\n' 2>/dev/null | \
    sort -nr | head -n 1 | cut -d' ' -f2-)
if [[ -z "${PACKAGE}" ]]; then
    echo "[ERROR] the official build produced no .run package" >&2
    exit 1
fi

echo "[INFO] installing ${PACKAGE} into ${INSTALL_ROOT}"
"${PACKAGE}" --quiet --install-path="${INSTALL_ROOT}"

CUSTOM_ROOT="${INSTALL_ROOT}/vendors/${VENDOR}_transformer"
if [[ ! -d "${CUSTOM_ROOT}" ]]; then
    CUSTOM_ROOT=$(find "${INSTALL_ROOT}/vendors" -mindepth 1 -maxdepth 1 -type d \
        -name '*_transformer' -print | head -n 1)
fi
if [[ -z "${CUSTOM_ROOT}" || ! -f "${CUSTOM_ROOT}/bin/set_env.bash" ]]; then
    echo "[ERROR] installed custom FAG tree was not found under ${INSTALL_ROOT}" >&2
    exit 1
fi

source "${CUSTOM_ROOT}/bin/set_env.bash"

ARCH=$(uname -m)
SAMPLE="${OPS_ROOT}/attention/flash_attention_score_grad/examples/test_aclnn_flash_attention_score_grad_v2.cpp"
BIN="${BUILD_ROOT}/fag_official_v2"

echo "[INFO] compiling one official FAG V2 example against libcust_opapi.so"
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

echo "[INFO] running the official FAG V2 example on NPU"
set +e
"${BIN}" 2>&1 | tee "${RESULT_ROOT}/run.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

if [[ ${RUN_STATUS} -eq 0 ]]; then
    echo "PASS" > "${RESULT_ROOT}/status.txt"
    echo "[PASS] official FAG ${TAG} completed; output values are in ${RESULT_ROOT}/run.log"
else
    echo "FAIL (${RUN_STATUS})" > "${RESULT_ROOT}/status.txt"
    echo "[FAIL] official FAG ${TAG} exited with ${RUN_STATUS}" >&2
fi

exit "${RUN_STATUS}"
