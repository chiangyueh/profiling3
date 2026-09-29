#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${FA_BACKWARD_RUNNER:-}" ]]; then
    echo "FA_BACKWARD_RUNNER must point to a FlashAttentionScoreGrad runner" >&2
    exit 2
fi

export ATTENTION_OPERATOR=flash_attention_score_grad
if [[ -x "${FA_BACKWARD_RUNNER}" ]]; then
    exec "${FA_BACKWARD_RUNNER}" "$@"
fi
exec bash "${FA_BACKWARD_RUNNER}" "$@"
