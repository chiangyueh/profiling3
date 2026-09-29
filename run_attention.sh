#!/bin/bash
set -euo pipefail

CURRENT_DIR=$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)
cd "${CURRENT_DIR}"

RUN_MODE="sim"
CYCLES_ONLY=0
SHORT=r:,v:
LONG=run-mode:,soc-version:,cycles-only
OPTS=$(getopt -a --options "${SHORT}" --longoptions "${LONG}" -- "$@")
eval set -- "${OPTS}"

while :; do
    case "$1" in
    -r | --run-mode)
        RUN_MODE="$2"
        shift 2
        ;;
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    --cycles-only)
        CYCLES_ONLY=1
        shift
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR]: Unexpected option: $1" >&2
        exit 1
        ;;
    esac
done

if [[ "${ATTENTION_GRAD:-0}" == "1" ]]; then
    OPERATOR="flash_attention_score_grad"
    OPERATOR_RUNNER="${FA_BACKWARD_RUNNER:-}"
else
    OPERATOR="flash_attention_score"
    OPERATOR_RUNNER="${FA_FORWARD_RUNNER:-}"
fi

if [[ -z "${OPERATOR_RUNNER}" ]]; then
    echo "[ERROR]: no real ${OPERATOR} runner is configured" >&2
    exit 2
fi

RUN_ARGS=(-r "${RUN_MODE}")
if [[ "${CYCLES_ONLY}" == "1" ]]; then
    RUN_ARGS+=(--cycles-only)
fi

export ATTENTION_OPERATOR="${OPERATOR}"
if [[ -x "${OPERATOR_RUNNER}" ]]; then
    exec "${OPERATOR_RUNNER}" "${RUN_ARGS[@]}"
fi
exec bash "${OPERATOR_RUNNER}" "${RUN_ARGS[@]}"
