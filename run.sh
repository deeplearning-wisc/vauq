#!/bin/bash
# Run VAUQ with LLaVA 1.5. topk_ratio and alpha are required.

set -euo pipefail

BENCHMARK="${1:?benchmark required (MMVet, CVBench, VILP, or VisualCot)}"
TOPK="${2:?topk_ratio required (e.g. 0.5)}"
ALPHA="${3:?alpha required (e.g. 1.0)}"
GPU_ID="${4:-0}"
MAX_SAMPLES="${5:-}"

CMD=(python run_vauq.py --model llava-1.5-7b-hf --benchmark "${BENCHMARK}" --topk_ratio "${TOPK}" --alpha "${ALPHA}")

if [[ -n "${MAX_SAMPLES}" ]]; then
    CMD+=(--max_samples "${MAX_SAMPLES}")
fi

echo "Running VAUQ | benchmark=${BENCHMARK} | topk=${TOPK} | alpha=${ALPHA} | GPU=${GPU_ID}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
