# VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation

Official code release for [**VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation**](https://arxiv.org/abs/2602.21054) (ACL 2026 Findings).

VAUQ is a **training-free** uncertainty score for large vision-language models. It combines:
1. **Predictive entropy** — standard token-level uncertainty of the generated answer.
2. **Image-Information Score (IS)** — how much uncertainty increases when salient vision tokens are masked (core-region masking via attention).

Final score (lower ⇒ more likely correct):

```
VAUQ = H(Y|X,V) − α · IS
IS   = H(Y|X,V_masked) − H(Y|X,V)
```



## Setup

```bash
conda create -n vauq python=3.10 -y
conda activate vauq
pip install -r requirements.txt
```

You need a CUDA GPU (~16 GB VRAM for LLaVA 1.5 7B). Models and datasets are downloaded automatically from Hugging Face on first run. For ViLP, log in with `huggingface-cli login` or set `HF_TOKEN` if the dataset requires access.

## Quick start

```bash
# Generate answers
python run_vauq.py --benchmark MMVet --generate_only

# Run VAUQ (uses benchmark defaults for topk_ratio and alpha)
python run_vauq.py --benchmark MMVet
./run.sh MMVet
./run.sh CVBench
./run.sh VILP

# Override hyperparameters
python run_vauq.py --benchmark MMVet --topk_ratio 0.5 --alpha 0.8

# Smoke test
CUDA_VISIBLE_DEVICES=0 python run_vauq.py --benchmark MMVet --max_samples 10
```

Results are written to `outputs/` as JSON, including per-sample VAUQ scores. AUROC/AUPR are computed only when you supply correctness labels (see below).


## Answer cache and labeling

This repo **does not** auto-label responses. VAUQ scoring only needs `ans` and `generated_ids`; AUROC/AUPR need a boolean `label` per sample that you provide (e.g. via your own LLM judge API, exact match, or human review).

**Step 1 — generate answers**

```bash
python run_vauq.py --benchmark MMVet --generate_only
# writes cache/llava-1.5-7b-hf_MMVet_answers.json
```

**Step 2 — add labels externally**

Each cache entry should look like `examples/example_cache.json`:

```json
{
  "0": {
    "question": "...",
    "gt_ans": "red",
    "ans": "The car is red.",
    "generated_ids": [[450, 8942, 29889]],
    "label": true
  }
}
```

Example judge prompt (same idea as in the paper):

```
Ground truth: {gt_ans}. Model answer: {ans}.
Does the model answer match the ground truth? Reply with Correct or Wrong only.
```

**Step 3 — run VAUQ + metrics**

```bash
python run_vauq.py --benchmark MMVet \
    --cache_path cache/llava-1.5-7b-hf_MMVet_answers.json
```

Re-runs reuse the labeled cache so you can ablate hyperparameters without re-generating text.

## Project structure

```
VAUQ/
├── run_vauq.py          # Main evaluation script
├── run.sh               # Convenience launcher
├── lvlm/llava.py        # LLaVA 1.5 + core attention masking
├── vauq/scoring.py      # Entropy, IS, and VAUQ score
├── benchmark/           # MM-Vet, CV-Bench & ViLP loaders
└── metrics.py           # Token-level entropy
```

## Citation

```bibtex
@article{park2026vauq,
  title={VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation},
  author={Park, Seongheon and Oh, Changdae and Choi, Hyeong Kyu and Du, Sean and Li, Sharon},
  journal={arXiv preprint arXiv:2602.21054},
  year={2026}
}
```

