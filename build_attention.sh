#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OPS_ROOT=${OPS_TRANSFORMER_ROOT:-"${SCRIPT_DIR}/../ops-transformer"}
ASCEND_ROOT=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
SOC_UNIT=${ATTENTION_SOC_UNIT:-ascend910b}
JOBS=${ATTENTION_BUILD_JOBS:-1}
INSTALL_ROOT=${ATTENTION_OPP_INSTALL_ROOT:-"${SCRIPT_DIR}/out/attention_opp"}
PATCH_FILE="${SCRIPT_DIR}/patches/ops_transformer_attention_search.patch"

if [[ ! -f "${OPS_ROOT}/build.sh" ]]; then
    echo "[ERROR] ops-transformer build.sh not found: ${OPS_ROOT}" >&2
    exit 2
fi

if git -C "${OPS_ROOT}" apply --unidiff-zero --check "${PATCH_FILE}" 2>/dev/null; then
    git -C "${OPS_ROOT}" apply --unidiff-zero "${PATCH_FILE}"
    echo "[INFO] applied FA/FAG search-parameter hook"
elif git -C "${OPS_ROOT}" apply --unidiff-zero --reverse --check "${PATCH_FILE}" 2>/dev/null; then
    echo "[INFO] FA/FAG search-parameter hook is already applied"
else
    echo "[ERROR] the FA/FAG patch does not match this ops-transformer revision" >&2
    exit 1
fi

(
    cd "${OPS_ROOT}"
    bash build.sh -j"${JOBS}" \
        --ops=flash_attention_score,flash_attention_score_grad \
        --soc="${SOC_UNIT}" --ophost --opapi --opkernel --pkg --incremental
)

PACKAGE=$(find "${OPS_ROOT}/output" "${OPS_ROOT}/build" -type f \
    -name 'cann-ops-transformer-*-linux-*.run' -printf '%T@ %p\n' 2>/dev/null | \
    sort -nr | head -n 1 | cut -d' ' -f2-)
if [[ -z "${PACKAGE}" ]]; then
    echo "[ERROR] custom operator .run package was not produced" >&2
    exit 1
fi

mkdir -p "${INSTALL_ROOT}"
"${PACKAGE}" --quiet --install-path="${INSTALL_ROOT}"

CUSTOM_ROOT=$(find "${INSTALL_ROOT}/vendors" -mindepth 1 -maxdepth 1 -type d \
    -name '*_transformer' -print | head -n 1)
if [[ -z "${CUSTOM_ROOT}" ]]; then
    echo "[ERROR] installed custom operator tree was not found under ${INSTALL_ROOT}" >&2
    exit 1
fi

CUSTOM_INCLUDE="${CUSTOM_ROOT}/op_api/include/aclnnop"
CUSTOM_LIBRARY="${CUSTOM_ROOT}/op_api/lib"
TOOLKIT_LIBRARY="${ASCEND_ROOT}/lib64"

g++ -O2 -std=c++17 "${SCRIPT_DIR}/attention_bench.cpp" \
    -I"${ASCEND_ROOT}/include" -I"${CUSTOM_INCLUDE}" \
    -L"${CUSTOM_LIBRARY}" -L"${TOOLKIT_LIBRARY}" \
    -Wl,-rpath,"${CUSTOM_LIBRARY}" -Wl,-rpath,"${TOOLKIT_LIBRARY}" \
    -lcust_opapi -lopapi_math -lascendcl -lnnopbase -lc_sec \
    -o "${SCRIPT_DIR}/attention_npu"

echo "[INFO] built ${SCRIPT_DIR}/attention_npu"
echo "[INFO] custom OPP: ${CUSTOM_ROOT}"
