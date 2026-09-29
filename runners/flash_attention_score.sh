#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${FA_FORWARD_RUNNER:-}" ]]; then
    echo "FA_FORWARD_RUNNER must point to a FlashAttentionScore runner" >&2
    exit 2
fi

export ATTENTION_OPERATOR=flash_attention_score
if [[ -x "${FA_FORWARD_RUNNER}" ]]; then
    exec "${FA_FORWARD_RUNNER}" "$@"
fi
exec bash "${FA_FORWARD_RUNNER}" "$@"
