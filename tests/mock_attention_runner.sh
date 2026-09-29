#!/usr/bin/env bash
set -euo pipefail

case "${ATTENTION_OPERATOR:-}" in
    flash_attention_score)
        : "${FA_S1_BASE:?}" "${FA_S2_BASE:?}" "${FA_D_BASE:?}" "${FA_CORE_NUM:?}"
        duration=$((FA_S1_BASE + FA_S2_BASE + FA_D_BASE + 1000 / FA_CORE_NUM))
        ;;
    flash_attention_score_grad)
        : "${FAG_S1_INNER:?}" "${FAG_S2_INNER:?}" "${FAG_S1_CV_RATIO:?}" "${FAG_S2_CV_RATIO:?}" "${FAG_CORE_NUM:?}"
        duration=$((FAG_S1_INNER + FAG_S2_INNER + FAG_S1_CV_RATIO + FAG_S2_CV_RATIO + 1000 / FAG_CORE_NUM))
        ;;
    *)
        echo "unknown ATTENTION_OPERATOR" >&2
        exit 2
        ;;
esac

echo "DURATION_US=${duration}"
