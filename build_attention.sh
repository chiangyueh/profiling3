#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OPS_ROOT=${OPS_TRANSFORMER_ROOT:-"${SCRIPT_DIR}/../ops-transformer-official-8.5.0"}
EXPECTED_COMMIT=6ead121aded45355043b502756b6592fd7c30b14
ASCEND_ROOT=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
SOC_UNIT=${ATTENTION_SOC_UNIT:-ascend910b}
JOBS=${ATTENTION_BUILD_JOBS:-1}
TILING_KEY=${ATTENTION_TILING_KEY:-74804}
CACHE_BASE=${ATTENTION_KERNEL_CACHE_ROOT:-"${SCRIPT_DIR}/out/attention_cache"}
CACHE_DIR="${CACHE_BASE}/${EXPECTED_COMMIT}/${SOC_UNIT}/flash_attention_score_grad/${TILING_KEY}"
INSTALL_ROOT="${CACHE_DIR}/opp"
BINARY="${CACHE_DIR}/attention_npu"
PATCH_FILE="${SCRIPT_DIR}/patches/ops_transformer_attention_search.patch"
CACHE_MANIFEST="${CACHE_DIR}/build_manifest.txt"
PATCH_ACTIVE=0

restore_official_source() {
    if [[ "${PATCH_ACTIVE}" == "1" ]] && \
       git -C "${OPS_ROOT}" apply --unidiff-zero --reverse --check "${PATCH_FILE}" 2>/dev/null; then
        git -C "${OPS_ROOT}" apply --unidiff-zero --reverse "${PATCH_FILE}"
        echo "[INFO] restored the pristine ops-transformer v8.5.0 source"
    fi
}

trap restore_official_source EXIT

if [[ ! -f "${OPS_ROOT}/build.sh" ]]; then
    echo "[ERROR] ops-transformer v8.5.0 build.sh not found: ${OPS_ROOT}" >&2
    exit 2
fi

if [[ ! "${TILING_KEY}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] ATTENTION_TILING_KEY must be a non-negative integer: ${TILING_KEY}" >&2
    exit 2
fi

if [[ "$(git -C "${OPS_ROOT}" rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
    echo "[ERROR] build_attention.sh requires ops-transformer v8.5.0 (${EXPECTED_COMMIT})" >&2
    echo "[ERROR] current checkout: ${OPS_ROOT} ($(git -C "${OPS_ROOT}" rev-parse HEAD))" >&2
    exit 2
fi

PATCH_SHA256=$(sha256sum "${PATCH_FILE}" | awk '{print $1}')
BENCH_SHA256=$(sha256sum "${SCRIPT_DIR}/attention_bench.cpp" | awk '{print $1}')
EXPECTED_FINGERPRINT=$(printf '%s\n' \
    "schema=1" \
    "commit=${EXPECTED_COMMIT}" \
    "soc=${SOC_UNIT}" \
    "operator=flash_attention_score_grad" \
    "tiling_key=${TILING_KEY}" \
    "patch=${PATCH_SHA256}" \
    "bench=${BENCH_SHA256}")
CACHED_CUSTOM_ROOT=$(find "${INSTALL_ROOT}/vendors" -mindepth 1 -maxdepth 1 -type d \
    -name '*_transformer' -print 2>/dev/null | head -n 1 || true)

if [[ -x "${BINARY}" && \
      -n "${CACHED_CUSTOM_ROOT}" && \
      -f "${CACHED_CUSTOM_ROOT}/bin/set_env.bash" && \
      -f "${CACHED_CUSTOM_ROOT}/op_api/include/aclnnop/aclnn_flash_attention_score_grad.h" && \
      -f "${CACHED_CUSTOM_ROOT}/op_api/lib/libcust_opapi.so" && \
      -f "${CACHE_MANIFEST}" && \
      "$(<"${CACHE_MANIFEST}")" == "${EXPECTED_FINGERPRINT}" ]]; then
    echo "[INFO] cache hit: FlashAttentionScoreGrad tiling key ${TILING_KEY}"
    echo "[INFO] reusing ${BINARY}"
    exit 0
fi

echo "[INFO] cache miss: compiling FlashAttentionScoreGrad tiling key ${TILING_KEY} once"

if git -C "${OPS_ROOT}" apply --unidiff-zero --check "${PATCH_FILE}" 2>/dev/null; then
    git -C "${OPS_ROOT}" apply --unidiff-zero "${PATCH_FILE}"
    PATCH_ACTIVE=1
    echo "[INFO] applied FA/FAG search-parameter hook"
elif git -C "${OPS_ROOT}" apply --unidiff-zero --reverse --check "${PATCH_FILE}" 2>/dev/null; then
    PATCH_ACTIVE=1
    echo "[INFO] FA/FAG search-parameter hook is already applied"
else
    echo "[ERROR] the FA/FAG patch does not match this ops-transformer revision" >&2
    exit 1
fi

(
    cd "${OPS_ROOT}"
    bash build.sh -j"${JOBS}" \
        --ops=flash_attention_score_grad \
        --soc="${SOC_UNIT}" --tiling_key="${TILING_KEY}" --pkg
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

g++ -O2 -std=c++17 -DATTENTION_GRAD_ONLY "${SCRIPT_DIR}/attention_bench.cpp" \
    -I"${ASCEND_ROOT}/include" -I"${CUSTOM_INCLUDE}" \
    -L"${CUSTOM_LIBRARY}" -L"${TOOLKIT_LIBRARY}" \
    -Wl,-rpath,"${CUSTOM_LIBRARY}" -Wl,-rpath,"${TOOLKIT_LIBRARY}" \
    -lcust_opapi -lopapi_math -lascendcl -lnnopbase -lc_sec \
    -o "${BINARY}"

mkdir -p "${INSTALL_ROOT}"
printf '%s\n' "${EXPECTED_FINGERPRINT}" > "${CACHE_MANIFEST}"

echo "[INFO] built ${BINARY}"
echo "[INFO] custom OPP: ${CUSTOM_ROOT}"
