#!/bin/bash
# NEW BEGIN
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OPS_ROOT=${OPS_TRANSFORMER_ROOT:-"${SCRIPT_DIR}/../ops-transformer-official-8.5.0"}
EXPECTED_COMMIT=6ead121aded45355043b502756b6592fd7c30b14
ASCEND_ROOT=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
SOC_UNIT=${FA_SOC_UNIT:-ascend910b}
JOBS=${FA_BUILD_JOBS:-1}
TILING_KEY=${FA_TILING_KEY:-1144808752}
CACHE_BASE=${FA_CACHE_ROOT:-"${SCRIPT_DIR}/out/fa_cache"}
CACHE_DIR="${CACHE_BASE}/${EXPECTED_COMMIT}/${SOC_UNIT}/${TILING_KEY}"
INSTALL_ROOT="${CACHE_DIR}/opp"
BINARY="${CACHE_DIR}/fa_npu"
PATCH_FILE="${SCRIPT_DIR}/patches/fa_candidate_hook.patch"
BUILD_PATCH_FILE="${SCRIPT_DIR}/patches/ops_transformer_incremental_build.patch"
CACHE_MANIFEST="${CACHE_DIR}/build_manifest.txt"
PACKAGE_MANIFEST="${OPS_ROOT}/build/fa_package_manifest.txt"
BUILD_LOG="${CACHE_DIR}/build.log"
INSTALL_LOG="${CACHE_DIR}/install.log"
PATCH_ACTIVE=0
BUILD_PATCH_ACTIVE=0

latest_package() {
    local roots=()
    local root
    for root in "${OPS_ROOT}/output" "${OPS_ROOT}/build" "${OPS_ROOT}/build_out"; do
        [[ -d "${root}" ]] && roots+=("${root}")
    done
    if [[ ${#roots[@]} -eq 0 ]]; then
        return 0
    fi
    find "${roots[@]}" -type f \
        \( -name 'cann-ops-transformer-*_linux-*.run' -o \
           -name 'cann-ops-transformer-*-linux-*.run' \) \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

restore_source() {
    if [[ "${PATCH_ACTIVE}" == "1" ]] && \
       git -C "${OPS_ROOT}" apply --unidiff-zero --reverse --check "${PATCH_FILE}" 2>/dev/null; then
        git -C "${OPS_ROOT}" apply --unidiff-zero --reverse "${PATCH_FILE}"
    fi
    if [[ "${BUILD_PATCH_ACTIVE}" == "1" ]] && \
       git -C "${OPS_ROOT}" apply --reverse --check "${BUILD_PATCH_FILE}" 2>/dev/null; then
        git -C "${OPS_ROOT}" apply --reverse "${BUILD_PATCH_FILE}"
    fi
}

trap restore_source EXIT

if [[ ! -f "${OPS_ROOT}/build.sh" ]]; then
    echo "ops-transformer v8.5.0 not found: ${OPS_ROOT}" >&2
    exit 2
fi
if [[ ! "${TILING_KEY}" =~ ^[0-9]+$ ]]; then
    echo "invalid FA_TILING_KEY: ${TILING_KEY}" >&2
    exit 2
fi
if [[ "$(git -C "${OPS_ROOT}" rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
    echo "ops-transformer must be at ${EXPECTED_COMMIT}: ${OPS_ROOT}" >&2
    exit 2
fi

PATCH_SHA256=$(sha256sum "${PATCH_FILE}" | awk '{print $1}')
BUILD_PATCH_SHA256=$(sha256sum "${BUILD_PATCH_FILE}" | awk '{print $1}')
LAUNCHER_SHA256=$(sha256sum "${SCRIPT_DIR}/fa_launcher.cpp" | awk '{print $1}')
PACKAGE_FINGERPRINT=$(printf '%s\n' \
    "schema=1" \
    "commit=${EXPECTED_COMMIT}" \
    "soc=${SOC_UNIT}" \
    "tiling_key=${TILING_KEY}" \
    "patch=${PATCH_SHA256}" \
    "build_patch=${BUILD_PATCH_SHA256}")
FULL_FINGERPRINT=$(printf '%s\n' "${PACKAGE_FINGERPRINT}" "launcher=${LAUNCHER_SHA256}")
CACHED_CUSTOM_ROOT=$(find "${INSTALL_ROOT}/vendors" -mindepth 1 -maxdepth 1 -type d \
    -name '*_transformer' -print 2>/dev/null | head -n 1 || true)

if [[ -x "${BINARY}" && -n "${CACHED_CUSTOM_ROOT}" && \
      -f "${CACHED_CUSTOM_ROOT}/bin/set_env.bash" && \
      -f "${CACHED_CUSTOM_ROOT}/op_api/include/aclnnop/aclnn_flash_attention_score.h" && \
      -f "${CACHED_CUSTOM_ROOT}/op_api/lib/libcust_opapi.so" && \
      -f "${CACHE_MANIFEST}" && "$(<"${CACHE_MANIFEST}")" == "${FULL_FINGERPRINT}" ]]; then
    echo "BUILD CACHE HIT: tiling_key=${TILING_KEY}"
    exit 0
fi

CACHED_PACKAGE_FINGERPRINT=""
if [[ -f "${CACHE_MANIFEST}" ]]; then
    CACHED_PACKAGE_FINGERPRINT=$(grep -v '^launcher=' "${CACHE_MANIFEST}" || true)
fi

if [[ -n "${CACHED_CUSTOM_ROOT}" && \
      -f "${CACHED_CUSTOM_ROOT}/bin/set_env.bash" && \
      -f "${CACHED_CUSTOM_ROOT}/op_api/include/aclnnop/aclnn_flash_attention_score.h" && \
      -f "${CACHED_CUSTOM_ROOT}/op_api/lib/libcust_opapi.so" && \
      "${CACHED_PACKAGE_FINGERPRINT}" == "${PACKAGE_FINGERPRINT}" ]]; then
    CUSTOM_ROOT="${CACHED_CUSTOM_ROOT}"
else
    PACKAGE=""
    if [[ -f "${PACKAGE_MANIFEST}" && "$(<"${PACKAGE_MANIFEST}")" == "${PACKAGE_FINGERPRINT}" ]]; then
        PACKAGE=$(latest_package)
    fi
    if [[ -z "${PACKAGE}" ]]; then
        mkdir -p "${CACHE_DIR}"
        echo "BUILD START: tiling_key=${TILING_KEY} log=${BUILD_LOG}"
        if git -C "${OPS_ROOT}" apply --unidiff-zero --check "${PATCH_FILE}" 2>/dev/null; then
            git -C "${OPS_ROOT}" apply --unidiff-zero "${PATCH_FILE}"
            PATCH_ACTIVE=1
        elif git -C "${OPS_ROOT}" apply --unidiff-zero --reverse --check "${PATCH_FILE}" 2>/dev/null; then
            PATCH_ACTIVE=1
        else
            echo "FA candidate patch does not match ops-transformer v8.5.0" >&2
            exit 1
        fi
        if git -C "${OPS_ROOT}" apply --check "${BUILD_PATCH_FILE}" 2>/dev/null; then
            git -C "${OPS_ROOT}" apply "${BUILD_PATCH_FILE}"
            BUILD_PATCH_ACTIVE=1
        elif git -C "${OPS_ROOT}" apply --reverse --check "${BUILD_PATCH_FILE}" 2>/dev/null; then
            BUILD_PATCH_ACTIVE=1
        else
            echo "incremental build patch does not match ops-transformer v8.5.0" >&2
            exit 1
        fi
        if ! (
            cd "${OPS_ROOT}"
            export FA_INCREMENTAL_BUILD=1
            bash build.sh -j"${JOBS}" --ops=flash_attention_score \
                --soc="${SOC_UNIT}" --tiling_key="${TILING_KEY}" --pkg
        ) >"${BUILD_LOG}" 2>&1; then
            echo "build failed: ${BUILD_LOG}" >&2
            tail -n 40 "${BUILD_LOG}" >&2 || true
            exit 1
        fi
        PACKAGE=$(latest_package)
        printf '%s\n' "${PACKAGE_FINGERPRINT}" >"${PACKAGE_MANIFEST}"
    fi
    if [[ -z "${PACKAGE}" || ! -f "${PACKAGE}" ]]; then
        echo "custom operator package was not produced" >&2
        exit 1
    fi
    mkdir -p "${INSTALL_ROOT}"
    if ! "${PACKAGE}" --quiet --install-path="${INSTALL_ROOT}" >"${INSTALL_LOG}" 2>&1; then
        echo "package installation failed: ${INSTALL_LOG}" >&2
        tail -n 40 "${INSTALL_LOG}" >&2 || true
        exit 1
    fi
    CUSTOM_ROOT=$(find "${INSTALL_ROOT}/vendors" -mindepth 1 -maxdepth 1 -type d \
        -name '*_transformer' -print | head -n 1)
fi

CUSTOM_INCLUDE="${CUSTOM_ROOT}/op_api/include/aclnnop"
CUSTOM_LIBRARY="${CUSTOM_ROOT}/op_api/lib"
TOOLKIT_LIBRARY="${ASCEND_ROOT}/lib64"
mkdir -p "${CACHE_DIR}"
g++ -O2 -std=c++17 "${SCRIPT_DIR}/fa_launcher.cpp" \
    -I"${ASCEND_ROOT}/include" -I"${CUSTOM_INCLUDE}" \
    -L"${CUSTOM_LIBRARY}" -L"${TOOLKIT_LIBRARY}" \
    -Wl,-rpath,"${CUSTOM_LIBRARY}" -Wl,-rpath,"${TOOLKIT_LIBRARY}" \
    -lcust_opapi -lopapi_math -lascendcl -lnnopbase -lc_sec -ldl \
    -o "${BINARY}"
printf '%s\n' "${FULL_FINGERPRINT}" >"${CACHE_MANIFEST}"
echo "BUILD READY: tiling_key=${TILING_KEY}"
# NEW END
