#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OPS_ROOT=${OPS_TRANSFORMER_ROOT:-"$SCRIPT_DIR/../ops-transformer"}
OPERATOR=${1:-all}
LOG_PATH=${AUTOTILING_LOG:-"$SCRIPT_DIR/autotiling_run.log"}
OUTPUT_PATH=${AUTOTILING_OUTPUT:-"$SCRIPT_DIR/autotiling_results.json"}
PATCH_PATH="$SCRIPT_DIR/patches/ops_transformer_autotiling_dump.patch"

case "$OPERATOR" in
    forward)
        OPS=flash_attention_score
        DEFAULT_FILTER='*FlashAttentionScoreArch22TilingTest*'
        ;;
    backward)
        OPS=flash_attention_score_grad
        DEFAULT_FILTER='FlashAttentionScoreGradTiling.*'
        ;;
    all)
        OPS=flash_attention_score,flash_attention_score_grad
        DEFAULT_FILTER='*FlashAttentionScoreArch22TilingTest*:*FlashAttentionScoreGradTiling.*'
        ;;
    *)
        echo "usage: $0 [forward|backward|all]" >&2
        exit 2
        ;;
esac

if [[ ! -x "$OPS_ROOT/build.sh" ]]; then
    echo "ops-transformer build.sh not found under: $OPS_ROOT" >&2
    exit 2
fi

DUMP_SOURCE="$OPS_ROOT/tests/ut/framework_normal/common/tiling_case_executor.cpp"
if ! grep -q 'static void DumpAutotiling' "$DUMP_SOURCE"; then
    if ! git -C "$OPS_ROOT" apply --check "$PATCH_PATH"; then
        echo "autotiling dump patch does not match this ops-transformer revision" >&2
        exit 1
    fi
    git -C "$OPS_ROOT" apply "$PATCH_PATH"
    echo "applied autotiling dump hook to: $DUMP_SOURCE"
fi

(
    cd "$OPS_ROOT"
    bash build.sh -j8 -u --ophost --noexec --ops="$OPS"
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
