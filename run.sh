#!/bin/bash
# Run VAUQ with LLaVA 1.5 (benchmark default hyperparameters).
#
# Usage:
#   ./run.sh MMVet
#   ./run.sh MMVet 7                    # GPU 7
#   ./run.sh MMVet 0.4 0.6              # override topk_ratio and alpha
#   ./run.sh MMVet 0.4 0.6 7 20         # override + GPU 7 + 20 samples

set -euo pipefail

BENCHMARK="${1:?benchmark required (MMVet, CVBench, or VILP)}"

CMD=(python run_vauq.py --model llava-1.5-7b-hf --benchmark "${BENCHMARK}")

if [[ $# -ge 3 && "$2" == *.* ]]; then
    CMD+=(--topk_ratio "$2" --alpha "$3")
    GPU_ID="${4:-0}"
    MAX_SAMPLES="${5:-}"
elif [[ $# -ge 2 ]]; then
    GPU_ID="$2"
    MAX_SAMPLES="${3:-}"
else
    GPU_ID="0"
    MAX_SAMPLES=""
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
    CMD+=(--max_samples "${MAX_SAMPLES}")
fi

echo "Running VAUQ | benchmark=${BENCHMARK} | GPU=${GPU_ID}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
