#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OPS_ROOT=${OPS_TRANSFORMER_ROOT:-"${SCRIPT_DIR}/../ops-transformer-official-8.5.0"}
SOURCE_VERSION=${OPS_TRANSFORMER_VERSION:-v8.5.0}
ASCEND_ROOT=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
SOC_UNIT=${FA_SOC_UNIT:-ascend910b}
JOBS=${FA_BUILD_JOBS:-1}
TILING_KEY=${FA_TILING_KEY:-}
DTYPE_KIND=${FA_DTYPE_KIND:-}
DISCOVER_ONLY=${FA_DISCOVER_ONLY:-0}
CACHE_BASE=${FA_CACHE_ROOT:-"${SCRIPT_DIR}/out/fa_cache"}
if [[ "${DISCOVER_ONLY}" == "1" ]]; then
    BUILD_MODE=host
    CACHE_ID=host
else
    BUILD_MODE=kernel
    CACHE_ID="${TILING_KEY}/dtype_${DTYPE_KIND}"
fi
CACHE_DIR="${CACHE_BASE}/${SOURCE_VERSION}/${SOC_UNIT}/${CACHE_ID}"
INSTALL_ROOT="${CACHE_DIR}/opp"
BINARY="${CACHE_DIR}/fa_npu"
PATCH_FILE="${SCRIPT_DIR}/patches/fa_candidate_hook.patch"
BUILD_PATCH_FILE="${SCRIPT_DIR}/patches/ops_transformer_incremental_build.patch"
CACHE_MANIFEST="${CACHE_DIR}/build_manifest.txt"
BUILD_LOG="${CACHE_DIR}/build.log"
INSTALL_LOG="${CACHE_DIR}/install.log"
PACKAGE_OUTPUT="${OPS_ROOT}/build_out/cann-ops-transformer-custom_linux-$(uname -m).run"
PATCH_ACTIVE=0
BUILD_PATCH_ACTIVE=0

kernel_cache_valid() {
    local custom_root=$1
    if [[ "${BUILD_MODE}" == "host" ]]; then
        return 0
    fi
    python3 - "${custom_root}" "${SOC_UNIT}" "${TILING_KEY}" "${DTYPE_KIND}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
soc = sys.argv[2]
tiling_key = int(sys.argv[3])
dtype = {"0": "float32", "1": "float16", "2": "bfloat16"}[sys.argv[4]]
paths = root.glob(f"op_impl/ai_core/tbe/kernel/{soc}/flash_attention_score/*.json")
selected = None
for path in paths:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    keys = {int(item["tilingKey"]) for item in data.get("kernelList", []) if "tilingKey" in item}
    inputs = {item.get("name"): item for item in data.get("supportInfo", {}).get("inputs", [])}
    binary = path.with_name(data.get("binFileName", "") + data.get("binFileSuffix", ".o"))
    if tiling_key in keys and inputs.get("query", {}).get("dtype") == dtype and binary.is_file():
        selected = (path, data)
        break
if selected is None:
    raise SystemExit(1)
path, data = selected
config_root = root / "op_impl/ai_core/tbe/kernel/config" / soc
try:
    binary_info = json.loads((config_root / "binary_info_config.json").read_text())
    op_config = json.loads((config_root / "flash_attention_score.json").read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
relative = f"{soc}/flash_attention_score/{path.name}"
simplified = set(data.get("supportInfo", {}).get("simplifiedKey", []))
registered = any(
    item.get("jsonPath") == relative and simplified.intersection(item.get("simplifiedKey", []))
    for item in binary_info.get("FlashAttentionScore", {}).get("binaryList", [])
)
listed = any(
    item.get("binInfo", {}).get("jsonFilePath") == relative
    and simplified.intersection(item.get("simplifiedKey", []))
    for item in op_config.get("binList", [])
)
raise SystemExit(0 if registered and listed else 1)
PY
}

reset_kernel_target() {
    local binary_root="${OPS_ROOT}/build/binary/${SOC_UNIT}"
    local generated="${binary_root}/gen"
    local kernels="${binary_root}/bin/flash_attention_score"
    rm -f -- "${binary_root}/src/flash_attention_score/flash_attention_score_${SOC_UNIT}_src_copy.done"
    if [[ -d "${generated}" ]]; then
        find "${generated}" -maxdepth 1 -type f \
            -name 'flash_attention_score_*.done' -delete
    fi
    if [[ -d "${kernels}" ]]; then
        find "${kernels}" -maxdepth 1 -type f \
            \( -name 'FlashAttentionScore_*.json' -o -name 'FlashAttentionScore_*.o' \) -delete
    fi
    rm -f -- "${binary_root}/bin/binary_info_config.json" \
        "${binary_root}/bin/flash_attention_score.json"
}

restore_source() {
    if [[ "${PATCH_ACTIVE}" == "1" ]]; then
        (cd "${OPS_ROOT}" && git apply -R "${PATCH_FILE}") || true
    fi
    if [[ "${BUILD_PATCH_ACTIVE}" == "1" ]]; then
        (cd "${OPS_ROOT}" && git apply -R "${BUILD_PATCH_FILE}") || true
    fi
}

apply_source_patch() {
    local patch_file=$1
    local label=$2

    if (cd "${OPS_ROOT}" && git apply --check "${patch_file}") 2>/dev/null; then
        (cd "${OPS_ROOT}" && git apply "${patch_file}")
        return 0
    fi
    if (cd "${OPS_ROOT}" && git apply -R --check "${patch_file}") 2>/dev/null; then
        (cd "${OPS_ROOT}" && git apply -R "${patch_file}")
        (cd "${OPS_ROOT}" && git apply "${patch_file}")
        return 0
    fi
    echo "${label} does not match ops-transformer ${SOURCE_VERSION}" >&2
    return 1
}

trap restore_source EXIT

if [[ ! -f "${OPS_ROOT}/build.sh" ]]; then
    echo "ops-transformer v8.5.0 not found: ${OPS_ROOT}" >&2
    exit 2
fi
if [[ "${BUILD_MODE}" == "kernel" && ! "${TILING_KEY}" =~ ^[0-9]+$ ]]; then
    echo "invalid FA_TILING_KEY: ${TILING_KEY}" >&2
    exit 2
fi
if [[ "${BUILD_MODE}" == "kernel" && ! "${DTYPE_KIND}" =~ ^[0-2]$ ]]; then
    echo "invalid FA_DTYPE_KIND: ${DTYPE_KIND}" >&2
    exit 2
fi
PATCH_SHA256=$(sha256sum "${PATCH_FILE}" | awk '{print $1}')
LAUNCHER_SHA256=$(sha256sum "${SCRIPT_DIR}/fa_launcher.cpp" | awk '{print $1}')
PACKAGE_FINGERPRINT=$(printf '%s\n' \
    "schema=3" \
    "version=${SOURCE_VERSION}" \
    "soc=${SOC_UNIT}" \
    "mode=${BUILD_MODE}" \
    "tiling_key=${TILING_KEY:-none}" \
    "dtype_kind=${DTYPE_KIND:-none}" \
    "patch=${PATCH_SHA256}")
if [[ -n "${FA_KERNEL_SOURCE_SHA256:-}" ]]; then
    PACKAGE_FINGERPRINT=$(printf '%s\n' "${PACKAGE_FINGERPRINT}" "kernel_source=${FA_KERNEL_SOURCE_SHA256}")
fi
FULL_FINGERPRINT=$(printf '%s\n' "${PACKAGE_FINGERPRINT}" "launcher=${LAUNCHER_SHA256}")
CACHED_CUSTOM_ROOT=$(find "${INSTALL_ROOT}/vendors" -mindepth 1 -maxdepth 1 -type d \
    -name '*_transformer' -print 2>/dev/null | head -n 1 || true)

if [[ -x "${BINARY}" && -n "${CACHED_CUSTOM_ROOT}" && \
      -f "${CACHED_CUSTOM_ROOT}/bin/set_env.bash" && \
      -f "${CACHED_CUSTOM_ROOT}/op_api/include/aclnnop/aclnn_flash_attention_score.h" && \
      -f "${CACHED_CUSTOM_ROOT}/op_api/lib/libcust_opapi.so" && \
      -f "${CACHE_MANIFEST}" && "$(<"${CACHE_MANIFEST}")" == "${FULL_FINGERPRINT}" ]] && \
      kernel_cache_valid "${CACHED_CUSTOM_ROOT}"; then
    echo "BUILD CACHE HIT: ${BUILD_MODE}=${CACHE_ID}"
    exit 0
fi

CACHED_PACKAGE_FINGERPRINT=""
if [[ -f "${CACHE_MANIFEST}" ]]; then
    CACHED_PACKAGE_FINGERPRINT=$(grep -v '^launcher=' "${CACHE_MANIFEST}" || true)
fi
if [[ -z "${FA_KERNEL_SOURCE_SHA256:-}" && -z "${CACHED_PACKAGE_FINGERPRINT}" && -n "${CACHED_CUSTOM_ROOT}" && \
      -f "${BUILD_LOG}" && -f "${INSTALL_LOG}" ]] && \
      kernel_cache_valid "${CACHED_CUSTOM_ROOT}"; then
    CACHED_PACKAGE_FINGERPRINT="${PACKAGE_FINGERPRINT}"
    echo "BUILD CACHE RECOVERED: ${BUILD_MODE}=${CACHE_ID}"
fi

if [[ -n "${CACHED_CUSTOM_ROOT}" && \
      -f "${CACHED_CUSTOM_ROOT}/bin/set_env.bash" && \
      -f "${CACHED_CUSTOM_ROOT}/op_api/include/aclnnop/aclnn_flash_attention_score.h" && \
      -f "${CACHED_CUSTOM_ROOT}/op_api/lib/libcust_opapi.so" && \
      "${CACHED_PACKAGE_FINGERPRINT}" == "${PACKAGE_FINGERPRINT}" ]] && \
      kernel_cache_valid "${CACHED_CUSTOM_ROOT}"; then
    CUSTOM_ROOT="${CACHED_CUSTOM_ROOT}"
else
    mkdir -p "${CACHE_DIR}"
    echo "BUILD START: ${BUILD_MODE}=${CACHE_ID} log=${BUILD_LOG}"
    apply_source_patch "${PATCH_FILE}" "FA candidate patch"
    PATCH_ACTIVE=1
    apply_source_patch "${BUILD_PATCH_FILE}" "incremental build patch"
    BUILD_PATCH_ACTIVE=1
    rm -f -- "${PACKAGE_OUTPUT}"
    if [[ "${BUILD_MODE}" == "kernel" ]]; then
        reset_kernel_target
    fi
    if ! (
        cd "${OPS_ROOT}"
        export ATTENTION_INCREMENTAL_BUILD=1
        if [[ "${BUILD_MODE}" == "host" ]]; then
            bash build.sh -j"${JOBS}" --ops=flash_attention_score \
                --soc="${SOC_UNIT}" --jit --pkg
        else
            bash build.sh -j"${JOBS}" --ops=flash_attention_score \
                --soc="${SOC_UNIT}" --tiling_key="${TILING_KEY}" --pkg
        fi
    ) >"${BUILD_LOG}" 2>&1; then
        echo "build failed: ${BUILD_LOG}" >&2
        tail -n 40 "${BUILD_LOG}" >&2 || true
        exit 1
    fi
    PACKAGE="${PACKAGE_OUTPUT}"
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
    if ! kernel_cache_valid "${CUSTOM_ROOT}"; then
        echo "built package is missing FlashAttentionScore tiling_key=${TILING_KEY} dtype_kind=${DTYPE_KIND}: ${PACKAGE}" >&2
        exit 1
    fi
fi

CUSTOM_INCLUDE="${CUSTOM_ROOT}/op_api/include/aclnnop"
CUSTOM_LIBRARY="${CUSTOM_ROOT}/op_api/lib"
TOOLKIT_LIBRARY="${ASCEND_ROOT}/lib64"
mkdir -p "${CACHE_DIR}"
g++ -O2 -std=c++17 "${SCRIPT_DIR}/fa_launcher.cpp" \
    -I"${ASCEND_ROOT}/include" -I"${CUSTOM_INCLUDE}" \
    -L"${CUSTOM_LIBRARY}" -L"${TOOLKIT_LIBRARY}" -L"${OPS_ROOT}/build" \
    -Wl,-rpath,"${CUSTOM_LIBRARY}" -Wl,-rpath,"${TOOLKIT_LIBRARY}" \
    -lcust_opapi -lopapi_math -lascendcl -lnnopbase -lc_sec -ldl \
    -o "${BINARY}"
printf '%s\n' "${FULL_FINGERPRINT}" >"${CACHE_MANIFEST}"
echo "BUILD READY: ${BUILD_MODE}=${CACHE_ID}"
