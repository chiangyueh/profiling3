#!/usr/bin/env bash
set -Eeuo pipefail

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
FORCE_REBUILD=${FAG_FORCE_REBUILD:-0}
CANN_LOG_LEVEL=${FAG_CANN_LOG_LEVEL:-${ASCEND_GLOBAL_LOG_LEVEL:-1}}
CANN_LOG_STDOUT=${FAG_CANN_LOG_STDOUT:-1}

SNAPSHOT="${SCRIPT_DIR}/vendor/ops-transformer-v8.5.0/attention/flash_attention_score_grad"
DEFAULT_OPS_ROOT="${SCRIPT_DIR}/../ops-transformer-official-8.5.0"
OPS_ROOT=${OPS_TRANSFORMER_ROOT:-${DEFAULT_OPS_ROOT}}
BUILD_ROOT=${FAG_BUILD_ROOT:-"${SCRIPT_DIR}/out/fag_npu"}
RESULT_ROOT=${FAG_RESULT_ROOT:-"${SCRIPT_DIR}/results/fag_npu"}
INSTALL_ROOT="${BUILD_ROOT}/opp"
LOG_ROOT="${RESULT_ROOT}/logs"
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_LOG="${LOG_ROOT}/run_npu_${RUN_ID}.log"
FULL_LOG="${RESULT_ROOT}/full.log"
STATUS_FILE="${RESULT_ROOT}/status.txt"
CURRENT_PHASE=initialization
CURRENT_PHASE_LOG=
LAST_ERROR=
ERROR_PATTERN='\[ERROR\]|fatal:|undefined reference|undefined symbol|not found|missing|No such file|failed|failure|cannot|invalid|Traceback|Exception|TIMEOUT'

mkdir -p "${RESULT_ROOT}" "${BUILD_ROOT}" "${INSTALL_ROOT}" "${LOG_ROOT}"
: > "${FULL_LOG}"
exec 3>&1 4>&2
exec > >(tee -a "${RUN_LOG}" "${FULL_LOG}" >&3) 2>&1
LOGGER_PID=$!
LOGS_FINALIZED=0

finalize_logs() {
    if [[ "${LOGS_FINALIZED}" == "1" ]]; then
        return 0
    fi
    LOGS_FINALIZED=1
    exec 1>&3 2>&4
    exec 3>&- 4>&-
    wait "${LOGGER_PID}" || true
}

print_error_lines() {
    local log_file="$1"
    if [[ -z "${log_file}" ]]; then
        return 0
    fi
    if [[ ! -s "${log_file}" ]]; then
        echo "  phase_log: ${log_file} (empty or missing)"
        return 0
    fi
    local matches
    matches=$(grep -Eai "${ERROR_PATTERN}" "${log_file}" | tail -40 || true)
    if [[ -n "${matches}" ]]; then
        echo "${matches}"
    else
        echo "  no matched error line; last 40 phase-log lines:"
        tail -40 "${log_file}" || true
    fi
    echo "  phase_log: ${log_file}"
}

print_runtime_diagnostics() {
    local log_file="$1"
    echo "  runtime_diagnostics:"
    if [[ ! -s "${log_file}" ]]; then
        echo "    no runtime output was produced before timeout"
        return 0
    fi
    local matches
    matches=$(grep -Eai \
        'FAGTiling|tiling.?key|FlashAttention|workspace|aclrt|kernel|error|failed|exception|507[0-9]+' \
        "${log_file}" | tail -100 || true)
    if [[ -n "${matches}" ]]; then
        echo "${matches}"
    else
        echo "    no matching CANN diagnostic line; last 80 runtime lines:"
        tail -80 "${log_file}" || true
    fi
}

report_failure() {
    local rc="$1"
    local line="$2"
    local command="$3"
    printf 'FAIL phase=%s exit_code=%s\n' "${CURRENT_PHASE}" "${rc}" > "${STATUS_FILE}"
    echo
    echo "FAG NPU run failed"
    echo "  phase:      ${CURRENT_PHASE}"
    echo "  exit_code:  ${rc}"
    echo "  line:       ${line}"
    echo "  command:    ${command}"
    if [[ -n "${LAST_ERROR}" ]]; then
        echo "  detail:     ${LAST_ERROR}"
    fi
    print_error_lines "${CURRENT_PHASE_LOG}"
    if [[ "${CURRENT_PHASE}" == "single-shape NPU execution" ]]; then
        print_runtime_diagnostics "${CURRENT_PHASE_LOG}"
    fi
    echo "  full_log:   ${FULL_LOG}"
    echo "  run_log:    ${RUN_LOG}"
}

on_error() {
    local rc=$?
    local line="${1:-unknown}"
    local command="${2:-unknown}"
    trap - ERR
    report_failure "${rc}" "${line}" "${command}"
    finalize_logs
    exit "${rc}"
}

on_signal() {
    local rc="$1"
    local signal="$2"
    trap - ERR INT TERM
    report_failure "${rc}" "signal" "received ${signal}"
    finalize_logs
    exit "${rc}"
}

require_command() {
    local command="$1"
    if ! command -v "${command}" >/dev/null 2>&1; then
        LAST_ERROR="missing required command: ${command}"
        echo "[ERROR] ${LAST_ERROR}" >&2
        return 1
    fi
}

require_file() {
    local label="$1"
    local path="$2"
    if [[ ! -f "${path}" ]]; then
        LAST_ERROR="missing ${label}: ${path}"
        echo "[ERROR] ${LAST_ERROR}" >&2
        return 1
    fi
}

require_dir() {
    local label="$1"
    local path="$2"
    if [[ ! -d "${path}" ]]; then
        LAST_ERROR="missing ${label}: ${path}"
        echo "[ERROR] ${LAST_ERROR}" >&2
        return 1
    fi
}

fail() {
    LAST_ERROR="$*"
    echo "[ERROR] ${LAST_ERROR}" >&2
    return 1
}

run_phase() {
    local label="$1"
    local log_file="$2"
    shift 2
    CURRENT_PHASE="${label}"
    CURRENT_PHASE_LOG="${log_file}"
    LAST_ERROR=
    echo
    echo "[PHASE] ${label}"
    echo "[PHASE] log: ${log_file}"
    if "$@" 2>&1 | tee "${log_file}"; then
        echo "[PHASE] ${label}: ok"
        return 0
    else
        local rc=${PIPESTATUS[0]}
        echo "[PHASE] ${label}: failed (exit ${rc})"
        return "${rc}"
    fi
}

trap 'on_error "${LINENO}" "${BASH_COMMAND}"' ERR
trap 'on_signal 130 INT' INT
trap 'on_signal 143 TERM' TERM

CURRENT_PHASE=preflight
for command in git diff find sort sed awk grep tail tee timeout env; do
    require_command "${command}"
done

if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    fail "FAG_BUILD_JOBS must be a positive integer; got '${JOBS}'"
fi
if [[ ! "${RUN_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
    fail "FAG_RUN_TIMEOUT must be a positive integer in seconds; got '${RUN_TIMEOUT}'"
fi
if [[ "${FORCE_REBUILD}" != "0" && "${FORCE_REBUILD}" != "1" ]]; then
    fail "FAG_FORCE_REBUILD must be 0 or 1; got '${FORCE_REBUILD}'"
fi

echo "[INFO] source: ops-transformer ${TAG} (${COMMIT})"
echo "[INFO] operator: FlashAttentionScoreGrad"
echo "[INFO] one shape: ${SHAPE}"
echo "[INFO] one tiling key: ${TILING_KEY}"
echo "[INFO] build jobs: ${JOBS}"
echo "[INFO] CANN runtime log level: ${CANN_LOG_LEVEL} (stdout=${CANN_LOG_STDOUT})"
if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    echo "[INFO] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
    echo "[INFO] launcher logical device 0 maps to the first selected NPU"
else
    echo "[INFO] ASCEND_RT_VISIBLE_DEVICES is unset; launcher uses device 0"
fi
echo "[INFO] full log: ${FULL_LOG}"
echo "[INFO] timestamped log: ${RUN_LOG}"

if ! git -C "${OPS_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    OPS_ROOT="${BUILD_ROOT}/ops-transformer-${TAG}"
    if [[ -e "${OPS_ROOT}" ]] && ! git -C "${OPS_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        fail "${OPS_ROOT} exists but is not a Git repository"
    fi
    if [[ ! -d "${OPS_ROOT}/.git" ]]; then
        echo "[INFO] downloading the official ${TAG} source"
        run_phase "official source download" "${LOG_ROOT}/source_download_${RUN_ID}.log" \
            git clone --depth 1 --branch "${TAG}" \
                https://gitcode.com/cann/ops-transformer.git "${OPS_ROOT}"
    fi
fi

CURRENT_PHASE=preflight
CURRENT_PHASE_LOG=
require_dir "official source directory" "${OPS_ROOT}"
require_file "official build script" "${OPS_ROOT}/build.sh"
require_file "official FAG launcher" \
    "${OPS_ROOT}/attention/flash_attention_score_grad/examples/test_aclnn_flash_attention_score_grad_v2.cpp"
require_file "official FAG host tiling test" \
    "${OPS_ROOT}/attention/flash_attention_score_grad/tests/ut/op_host/arch32/test_flash_attention_score_grad_tiling.cpp"
require_dir "vendored FAG source snapshot" "${SNAPSHOT}"

ACTUAL_COMMIT=$(git -C "${OPS_ROOT}" rev-parse HEAD)
if [[ "${ACTUAL_COMMIT}" != "${COMMIT}" ]]; then
    fail "expected ops-transformer ${COMMIT}, got ${ACTUAL_COMMIT}"
fi

if ! git -C "${OPS_ROOT}" diff --quiet -- attention/flash_attention_score_grad || \
   ! git -C "${OPS_ROOT}" diff --cached --quiet -- attention/flash_attention_score_grad; then
    fail "the official FAG source tree has tracked modifications: ${OPS_ROOT}/attention/flash_attention_score_grad"
fi

if ! diff -qr "${SNAPSHOT}" \
    "${OPS_ROOT}/attention/flash_attention_score_grad" >/dev/null; then
    fail "the build source does not match the vendored official FAG snapshot: ${SNAPSHOT}"
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
    fail "CANN environment script was not found under '${ASCEND_ROOT:-<unset>}'; set ASCEND_HOME_PATH"
fi

export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
CURRENT_PHASE="CANN environment setup"
CURRENT_PHASE_LOG=
source "${CANN_SETENV}"
ASCEND_ROOT=${ASCEND_HOME_PATH:-${ASCEND_ROOT}}
require_file "CANN ACL header" "${ASCEND_ROOT}/include/acl/acl.h"
CUSTOM_ROOT="${INSTALL_ROOT}/vendors/${VENDOR}_transformer"
ARCH=$(uname -m)
SAMPLE="${OPS_ROOT}/attention/flash_attention_score_grad/examples/test_aclnn_flash_attention_score_grad_v2.cpp"
BIN="${BUILD_ROOT}/fag_official_v2"
ARTIFACT_MANIFEST="${BUILD_ROOT}/artifact_manifest.txt"
EXPECTED_FINGERPRINT="commit=${COMMIT};soc=${SOC};key=${TILING_KEY};vendor=${VENDOR}"

require_file "CANN runtime library" "${ASCEND_ROOT}/lib64/libascendcl.so"
require_file "CANN NN operator base library" "${ASCEND_ROOT}/lib64/libnnopbase.so"
require_file "CANN secure C library" "${ASCEND_ROOT}/lib64/libc_sec.so"

MANIFEST_COMPATIBLE=0
if [[ -f "${ARTIFACT_MANIFEST}" ]]; then
    MANIFEST_CONTENT=$(<"${ARTIFACT_MANIFEST}")
    if [[ "${MANIFEST_CONTENT}" == *"commit=${COMMIT}"* && \
          "${MANIFEST_CONTENT}" == *"soc=${SOC}"* && \
          "${MANIFEST_CONTENT}" == *"key=${TILING_KEY}"* && \
          "${MANIFEST_CONTENT}" == *"vendor=${VENDOR}"* ]]; then
        MANIFEST_COMPATIBLE=1
    fi
fi

PACKAGE_REBUILD=0
if [[ "${FORCE_REBUILD}" == "1" ]]; then
    PACKAGE_REBUILD=1
    PACKAGE_CACHE_REASON="FAG_FORCE_REBUILD=1"
elif [[ ! -f "${CUSTOM_ROOT}/bin/set_env.bash" ]]; then
    PACKAGE_REBUILD=1
    PACKAGE_CACHE_REASON="installed custom environment is missing"
elif [[ ! -f "${CUSTOM_ROOT}/op_api/include/aclnnop/aclnn_flash_attention_score_grad.h" ]]; then
    PACKAGE_REBUILD=1
    PACKAGE_CACHE_REASON="installed FAG API header is missing"
elif [[ ! -f "${CUSTOM_ROOT}/op_api/lib/libcust_opapi.so" ]]; then
    PACKAGE_REBUILD=1
    PACKAGE_CACHE_REASON="installed FAG API library is missing"
elif [[ -f "${ARTIFACT_MANIFEST}" && "${MANIFEST_COMPATIBLE}" != "1" ]]; then
    PACKAGE_REBUILD=1
    PACKAGE_CACHE_REASON="source commit, SoC, tiling key, or vendor changed"
elif [[ ! -f "${ARTIFACT_MANIFEST}" ]]; then
    PACKAGE_CACHE_REASON="complete pre-manifest one-key package found; adopting it"
else
    PACKAGE_CACHE_REASON="matching one-key package already exists"
fi

if [[ "${PACKAGE_REBUILD}" == "0" ]]; then
    echo "[INFO] package cache hit: ${PACKAGE_CACHE_REASON}"
    echo "[INFO] skipping FAG kernel package build and installation"
else
    echo "[INFO] package cache miss: ${PACKAGE_CACHE_REASON}"
    require_command bisheng

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
            require_command python3
            run_phase "local CMake install" "${LOG_ROOT}/cmake_install_${RUN_ID}.log" \
                python3 -m pip install --no-cache-dir --target "${CMAKE_ROOT}" cmake==3.28.3
        fi
        export PATH="${CMAKE_BIN}:${PATH}"
    fi

    require_command cmake
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
    build_one_key_package() {
        cd "${OPS_ROOT}" || return $?
        "${BUILD_PREFIX[@]}" bash build.sh \
            --pkg \
            --ops=flash_attention_score_grad \
            --soc="${SOC}" \
            --vendor_name="${VENDOR}" \
            --tiling_key="${TILING_KEY}" \
            -j"${JOBS}"
    }

    run_phase "one-key package build" "${LOG_ROOT}/package_build_${RUN_ID}.log" \
        build_one_key_package

    CURRENT_PHASE="package discovery"
    CURRENT_PHASE_LOG="${LOG_ROOT}/package_build_${RUN_ID}.log"
    PACKAGE_ROOTS=()
    for package_dir in "${OPS_ROOT}/output" "${OPS_ROOT}/build"; do
        [[ -d "${package_dir}" ]] && PACKAGE_ROOTS+=("${package_dir}")
    done
    if [[ ${#PACKAGE_ROOTS[@]} -eq 0 ]]; then
        fail "build completed but no package directory exists under ${OPS_ROOT}"
    fi
    PACKAGE=$(find "${PACKAGE_ROOTS[@]}" -type f -name '*.run' -printf '%T@ %p\n' 2>/dev/null | \
        sort -nr | sed -n '1p' | cut -d' ' -f2- || true)
    if [[ -z "${PACKAGE}" ]]; then
        fail "build completed but no .run package was found under: ${PACKAGE_ROOTS[*]}"
    fi
    require_file "generated custom package" "${PACKAGE}"

    echo "[INFO] installing the one-key package into ${INSTALL_ROOT}"
    run_phase "custom package install" "${LOG_ROOT}/package_install_${RUN_ID}.log" \
        "${PACKAGE}" --quiet --install-path="${INSTALL_ROOT}"

    CURRENT_PHASE="installed package validation"
    CURRENT_PHASE_LOG="${LOG_ROOT}/package_install_${RUN_ID}.log"
    require_dir "installed custom FAG directory" "${CUSTOM_ROOT}"
    require_file "custom FAG environment script" "${CUSTOM_ROOT}/bin/set_env.bash"
    require_dir "custom FAG API include directory" "${CUSTOM_ROOT}/op_api/include"
    require_dir "custom FAG API library directory" "${CUSTOM_ROOT}/op_api/lib"
    require_file "custom FAG API header" \
        "${CUSTOM_ROOT}/op_api/include/aclnnop/aclnn_flash_attention_score_grad.h"
    require_file "custom FAG API library" "${CUSTOM_ROOT}/op_api/lib/libcust_opapi.so"
fi

CURRENT_PHASE="installed package validation"
CURRENT_PHASE_LOG=
require_file "custom FAG environment script" "${CUSTOM_ROOT}/bin/set_env.bash"
require_file "custom FAG API header" \
    "${CUSTOM_ROOT}/op_api/include/aclnnop/aclnn_flash_attention_score_grad.h"
require_file "custom FAG API library" "${CUSTOM_ROOT}/op_api/lib/libcust_opapi.so"
source "${CUSTOM_ROOT}/bin/set_env.bash"

LAUNCHER_REBUILD=0
if [[ "${FORCE_REBUILD}" == "1" ]]; then
    LAUNCHER_REBUILD=1
    LAUNCHER_CACHE_REASON="FAG_FORCE_REBUILD=1"
elif [[ "${PACKAGE_REBUILD}" == "1" ]]; then
    LAUNCHER_REBUILD=1
    LAUNCHER_CACHE_REASON="FAG package was rebuilt"
elif [[ ! -x "${BIN}" ]]; then
    LAUNCHER_REBUILD=1
    LAUNCHER_CACHE_REASON="compiled launcher is missing or not executable"
else
    LAUNCHER_CACHE_REASON="compiled launcher already exists"
fi

if [[ "${LAUNCHER_REBUILD}" == "0" ]]; then
    echo "[INFO] launcher cache hit: ${LAUNCHER_CACHE_REASON}"
    echo "[INFO] skipping launcher compilation"
else
    echo "[INFO] launcher cache miss: ${LAUNCHER_CACHE_REASON}"
    require_command g++
    echo "[INFO] compiling the official one-shape launcher"
    COMPILE_CMD=(g++ -O2 -std=c++17 "${SAMPLE}"
        -I"${ASCEND_ROOT}/include"
        -I"${CUSTOM_ROOT}/op_api/include"
        -I"${ASCEND_ROOT}/${ARCH}-linux/include/aclnnop"
        -L"${CUSTOM_ROOT}/op_api/lib"
        -L"${ASCEND_ROOT}/lib64"
        -Wl,-rpath,"${CUSTOM_ROOT}/op_api/lib"
        -Wl,-rpath,"${ASCEND_ROOT}/lib64"
        -lcust_opapi -lascendcl -lnnopbase -lpthread -lc_sec
        -o "${BIN}")
    run_phase "official launcher compile" "${LOG_ROOT}/launcher_compile_${RUN_ID}.log" \
        "${COMPILE_CMD[@]}"
    require_file "compiled FAG launcher" "${BIN}"
fi

CURRENT_PHASE="cached artifact validation"
CURRENT_PHASE_LOG=
require_file "custom FAG environment script" "${CUSTOM_ROOT}/bin/set_env.bash"
require_file "custom FAG API header" \
    "${CUSTOM_ROOT}/op_api/include/aclnnop/aclnn_flash_attention_score_grad.h"
require_file "custom FAG API library" "${CUSTOM_ROOT}/op_api/lib/libcust_opapi.so"
require_file "compiled FAG launcher" "${BIN}"

MANIFEST_TMP="${ARTIFACT_MANIFEST}.tmp.$$"
printf '%s\n' "${EXPECTED_FINGERPRINT}" > "${MANIFEST_TMP}"
mv "${MANIFEST_TMP}" "${ARTIFACT_MANIFEST}"
echo "[INFO] build artifacts ready: ${ARTIFACT_MANIFEST}"

echo "[INFO] running exactly one FAG shape on NPU (timeout: ${RUN_TIMEOUT}s)"
if run_phase "single-shape NPU execution" "${RESULT_ROOT}/run.log" \
    timeout --signal=INT --kill-after=10 "${RUN_TIMEOUT}" \
        env ASCEND_GLOBAL_LOG_LEVEL="${CANN_LOG_LEVEL}" \
            ASCEND_SLOG_PRINT_TO_STDOUT="${CANN_LOG_STDOUT}" \
            "${BIN}"; then
    echo "PASS" > "${STATUS_FILE}"
    echo "[PASS] one official FAG shape completed"
    echo "[PASS] output: ${RESULT_ROOT}/run.log"
else
    RUN_STATUS=$?
    if [[ ${RUN_STATUS} -eq 124 ]]; then
        LAST_ERROR="the one-shape NPU run exceeded ${RUN_TIMEOUT}s"
        echo "[ERROR] ${LAST_ERROR}" >&2
    fi
    report_failure "${RUN_STATUS}" "${LINENO}" "${BIN}"
    finalize_logs
    exit "${RUN_STATUS}"
fi

finalize_logs
