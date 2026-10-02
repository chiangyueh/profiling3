#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OPS_ROOT=${OPS_TRANSFORMER_ROOT:-"$SCRIPT_DIR/../ops-transformer-official-8.5.0"}
EXPECTED_COMMIT=6ead121aded45355043b502756b6592fd7c30b14
JOBS=${ATTENTION_BUILD_JOBS:-1}
MODE=${1:-}

SEARCH_PATCH="$SCRIPT_DIR/patches/ops_transformer_attention_search.patch"
BUILD_PATCH="$SCRIPT_DIR/patches/ops_transformer_host_jobs.patch"
FA_SOURCE="$SCRIPT_DIR/host_probe/test_flash_attention_score_search_tiling.cpp"
FAG_SOURCE="$SCRIPT_DIR/host_probe/test_flash_attention_score_grad_search_tiling.cpp"
FA_DEST="$OPS_ROOT/attention/flash_attention_score/tests/ut/op_host/arch32/test_flash_attention_score_search_tiling.cpp"
FAG_DEST="$OPS_ROOT/attention/flash_attention_score_grad/tests/ut/op_host/arch32/test_flash_attention_score_grad_search_tiling.cpp"
CACHE_ROOT=${ATTENTION_HOST_CACHE_ROOT:-"$SCRIPT_DIR/out/attention_host/$EXPECTED_COMMIT"}

case "$MODE" in
    build-forward|forward)
        OPERATOR=flash_attention_score
        PROBE_SOURCE="$FA_SOURCE"
        PROBE_DEST="$FA_DEST"
        FILTER=AttentionHostTiling.Forward
        ;;
    build-backward|backward)
        OPERATOR=flash_attention_score_grad
        PROBE_SOURCE="$FAG_SOURCE"
        PROBE_DEST="$FAG_DEST"
        FILTER=AttentionHostTiling.Backward
        ;;
    *)
        echo "usage: $0 {build-forward|forward|build-backward|backward}" >&2
        exit 2
        ;;
esac

CACHE_DIR="$CACHE_ROOT/$OPERATOR"
CACHE_BIN="$CACHE_DIR/transformer_op_host_ut"
CACHE_MANIFEST="$CACHE_DIR/build_manifest.txt"

if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] ATTENTION_BUILD_JOBS must be a positive integer: $JOBS" >&2
    exit 2
fi
if [[ ! -f "$OPS_ROOT/build.sh" ]]; then
    echo "[ERROR] ops-transformer v8.5.0 build.sh not found: $OPS_ROOT" >&2
    exit 2
fi
if [[ "$(git -C "$OPS_ROOT" rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
    echo "[ERROR] Host tiling requires ops-transformer v8.5.0 ($EXPECTED_COMMIT)" >&2
    echo "[ERROR] current checkout: $OPS_ROOT ($(git -C "$OPS_ROOT" rev-parse HEAD))" >&2
    exit 2
fi

FINGERPRINT=$(printf '%s\n' \
    "schema=4" \
    "commit=$EXPECTED_COMMIT" \
    "operator=$OPERATOR" \
    "search_patch=$(sha256sum "$SEARCH_PATCH" | awk '{print $1}')" \
    "build_patch=$(sha256sum "$BUILD_PATCH" | awk '{print $1}')" \
    "probe=$(sha256sum "$PROBE_SOURCE" | awk '{print $1}')")

SEARCH_APPLIED=0
BUILD_PATCH_APPLIED=0
PROBES_INSTALLED=0

restore_source() {
    if [[ "$PROBES_INSTALLED" == "1" ]]; then
        rm -f -- "$PROBE_DEST"
    fi
    if [[ "$BUILD_PATCH_APPLIED" == "1" ]] && \
       git -C "$OPS_ROOT" apply --reverse --check "$BUILD_PATCH" >/dev/null 2>&1; then
        git -C "$OPS_ROOT" apply --reverse "$BUILD_PATCH"
    fi
    if [[ "$SEARCH_APPLIED" == "1" ]] && \
       git -C "$OPS_ROOT" apply --unidiff-zero --reverse --check "$SEARCH_PATCH" >/dev/null 2>&1; then
        git -C "$OPS_ROOT" apply --unidiff-zero --reverse "$SEARCH_PATCH"
    fi
}

build_host_probe() {
    if [[ -x "$CACHE_BIN" && -f "$CACHE_MANIFEST" && \
          "$(<"$CACHE_MANIFEST")" == "$FINGERPRINT" ]]; then
        echo "[INFO] Host tiling cache hit: $CACHE_BIN"
        return
    fi

    echo "[INFO] Host tiling cache miss: building only $OPERATOR"
    echo "[INFO] This is CPU-only; it does not select or run an NPU"
    trap restore_source EXIT INT TERM

    if git -C "$OPS_ROOT" apply --unidiff-zero --check "$SEARCH_PATCH" >/dev/null 2>&1; then
        git -C "$OPS_ROOT" apply --unidiff-zero "$SEARCH_PATCH"
        SEARCH_APPLIED=1
    elif ! git -C "$OPS_ROOT" apply --unidiff-zero --reverse --check "$SEARCH_PATCH" >/dev/null 2>&1; then
        echo "[ERROR] attention search patch does not match the official checkout" >&2
        exit 1
    fi

    # Upstream's new UT path ignores build.sh -j and otherwise consumes every
    # CPU. Reuse the existing one-line build fix and keep ATTENTION_BUILD_JOBS=1.
    if ! grep -Fq 'cmake --build . --target ${UT_TARGET} ${JOB_NUM}' "$OPS_ROOT/build.sh"; then
        if git -C "$OPS_ROOT" apply --check "$BUILD_PATCH" >/dev/null 2>&1; then
            git -C "$OPS_ROOT" apply "$BUILD_PATCH"
            BUILD_PATCH_APPLIED=1
        else
            echo "[ERROR] low-resource Host UT build patch does not match the official checkout" >&2
            exit 1
        fi
    fi

    if [[ -e "$PROBE_DEST" ]] && ! cmp -s -- "$PROBE_SOURCE" "$PROBE_DEST"; then
        echo "[ERROR] a different file already exists at: $PROBE_DEST" >&2
        exit 1
    fi
    [[ -e "$PROBE_DEST" ]] || cp -- "$PROBE_SOURCE" "$PROBE_DEST"
    PROBES_INSTALLED=1

    (
        cd "$OPS_ROOT"
        bash build.sh -j"$JOBS" -u --ophost --noexec \
            --disable_asan --ccache false --soc=ascend910b --ops="$OPERATOR"
    )

    local built_bin
    built_bin=$(find "$OPS_ROOT/build" -type f -name transformer_op_host_ut -perm -111 | head -n 1)
    if [[ -z "$built_bin" ]]; then
        echo "[ERROR] transformer_op_host_ut was not produced" >&2
        exit 1
    fi
    mkdir -p "$CACHE_DIR"
    cp -- "$built_bin" "$CACHE_BIN"
    chmod +x "$CACHE_BIN"

    # The UT executable links the Host tiling implementation as a shared
    # library. Cache every dependency that resolves inside this build tree so
    # later single-key kernel builds cannot overwrite the Host probe runtime.
    declare -A copied_deps=()
    local dependency_queue=("$built_bin")
    local host_library
    host_library=$(find "$OPS_ROOT/build" -type f -name 'libophost_transformer_ut.so*' | head -n 1)
    if [[ -n "$host_library" ]]; then
        copied_deps[$host_library]=1
        cp -- "$host_library" "$CACHE_DIR/$(basename -- "$host_library")"
        dependency_queue+=("$host_library")
    fi
    local dependency_index=0
    while (( dependency_index < ${#dependency_queue[@]} )); do
        local object=${dependency_queue[$dependency_index]}
        dependency_index=$((dependency_index + 1))
        while IFS= read -r dependency; do
            [[ "$dependency" == "$OPS_ROOT/build/"* ]] || continue
            [[ -f "$dependency" ]] || continue
            [[ -z "${copied_deps[$dependency]:-}" ]] || continue
            copied_deps[$dependency]=1
            cp -- "$dependency" "$CACHE_DIR/$(basename -- "$dependency")"
            dependency_queue+=("$dependency")
        done < <(ldd "$object" 2>/dev/null | awk '/=> \/.*\// {print $3} /^\// {print $1}')
    done
    printf '%s\n' "$FINGERPRINT" > "$CACHE_MANIFEST"
    echo "[INFO] cached Host tiling executable: $CACHE_BIN"

    restore_source
    trap - EXIT INT TERM
}

build_host_probe

if [[ "$MODE" == build-* ]]; then
    exit 0
fi

LD_LIBRARY_PATH="$CACHE_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$CACHE_BIN" --gtest_filter="$FILTER"
