#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OPS_ROOT=${OPS_TRANSFORMER_ROOT:-"$SCRIPT_DIR/../ops-transformer-official-8.5.0"}
OPERATOR=${1:-backward}
LOG_PATH=${AUTOTILING_LOG:-"$SCRIPT_DIR/autotiling_run.log"}
OUTPUT_PATH=${AUTOTILING_OUTPUT:-"$SCRIPT_DIR/autotiling_results.json"}
PATCH_PATH="$SCRIPT_DIR/patches/ops_transformer_autotiling_dump.patch"
JOBS=${ATTENTION_BUILD_JOBS:-1}

case "$OPERATOR" in
    forward)
        OPS=flash_attention_score
        DEFAULT_FILTER='FlashAttentionScoreTiling.*'
        ;;
    backward)
        OPS=flash_attention_score_grad
        DEFAULT_FILTER='FlashAttentionScoreGradTiling.*'
        ;;
    all)
        OPS=flash_attention_score,flash_attention_score_grad
        DEFAULT_FILTER='FlashAttentionScoreTiling.*:FlashAttentionScoreGradTiling.*'
        ;;
    *)
        echo "usage: $0 [forward|backward|all]" >&2
        exit 2
        ;;
esac

if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ATTENTION_BUILD_JOBS must be a positive integer: ${JOBS}" >&2
    exit 2
fi

if [[ ! -f "$OPS_ROOT/build.sh" ]]; then
    echo "ops-transformer build.sh not found under: $OPS_ROOT" >&2
    exit 2
fi

DUMP_SOURCE="$OPS_ROOT/tests/ut/framework_normal/common/tiling_case_executor.cpp"
BUILD_SCRIPT="$OPS_ROOT/build.sh"
if grep -q 'static void DumpAutotiling' "$DUMP_SOURCE" &&
   grep -Fq 'cmake --build . --target ${UT_TARGET} ${JOB_NUM}' "$BUILD_SCRIPT"; then
    echo "autotiling dump hook is already applied"
elif grep -q 'static void DumpAutotiling' "$DUMP_SOURCE" ||
     grep -Fq 'cmake --build . --target ${UT_TARGET} ${JOB_NUM}' "$BUILD_SCRIPT"; then
    echo "autotiling patch is only partially applied; restore the official v8.5.0 checkout first" >&2
    exit 1
else
    if ! git -C "$OPS_ROOT" apply --check "$PATCH_PATH"; then
        echo "autotiling dump patch does not match this ops-transformer revision" >&2
        exit 1
    fi
    git -C "$OPS_ROOT" apply "$PATCH_PATH"
    echo "applied autotiling dump hook to: $DUMP_SOURCE"
fi

(
    cd "$OPS_ROOT"
    bash build.sh -j"${JOBS}" -u --ophost --noexec --ops="$OPS"
)

BIN=$(find "$OPS_ROOT/build" -type f -name transformer_op_host_ut -perm -111 | head -n 1)
if [[ -z "$BIN" ]]; then
    echo "transformer_op_host_ut was not produced under: $OPS_ROOT/build" >&2
    exit 1
fi

FILTER=${AUTOTILING_FILTER:-$DEFAULT_FILTER}
set +e
AUTOTILING_DUMP=1 "$BIN" --gtest_filter="$FILTER" 2>&1 | tee "$LOG_PATH"
TEST_STATUS=${PIPESTATUS[0]}
set -e

python3 "$SCRIPT_DIR/autotiling.py" "$LOG_PATH" --output "$OUTPUT_PATH"
echo "decoded autotiling: $OUTPUT_PATH"
exit "$TEST_STATUS"
