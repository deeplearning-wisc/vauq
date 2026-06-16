#!/usr/bin/env python3
"""
Run VAUQ self-evaluation on LLaVA 1.5.

Example:
    python run_vauq.py --benchmark MMVet --max_samples 10
    python run_vauq.py --benchmark MMVet --generate_only
"""

import argparse
import copy
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from benchmark import BENCHMARK_MAP
from lvlm.llava import LLaVA
from utils.misc import get_cur_time
from vauq.scoring import compute_vauq_scores

DEFAULT_HYPERPARAMETERS = {
    "llava-1.5-7b-hf": {
        "MMVet": {"topk_ratio": 0.4, "alpha": 0.6},
        "CVBench": {"topk_ratio": 0.1, "alpha": 1.5},
        "VILP": {"topk_ratio": 1.0, "alpha": 0.6},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="VAUQ on LLaVA 1.5")
    parser.add_argument(
        "--model",
        type=str,
        default="llava-1.5-7b-hf",
        help="HuggingFace model id suffix (llava-hf/<model>)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="MMVet",
        choices=list(BENCHMARK_MAP.keys()),
    )
    parser.add_argument("--inference_temp", type=float, default=0.1)
    parser.add_argument(
        "--cache_path",
        type=str,
        default=None,
        help="JSON cache with answers and user-provided labels (created if missing)",
    )
    parser.add_argument(
        "--generate_only",
        action="store_true",
        help="Only generate answers and save cache; skip VAUQ scoring",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for logs and metrics",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of benchmark samples (for quick tests)",
    )
    parser.add_argument(
        "--topk_ratio",
        type=float,
        default=None,
        help="Fraction of salient vision tokens to mask (default: benchmark default)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Weight for Image-Information Score (default: benchmark default)",
    )
    parser.add_argument(
        "--layer_start",
        type=int,
        default=10,
        help="First decoder layer for attention aggregation",
    )
    parser.add_argument(
        "--layer_end",
        type=int,
        default=25,
        help="Last decoder layer (exclusive) for attention aggregation",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_scoring_params(args):
    defaults = DEFAULT_HYPERPARAMETERS.get(args.model, {}).get(args.benchmark)
    if defaults is None:
        raise SystemExit(
            f"Error: no default hyperparameters for model={args.model}, "
            f"benchmark={args.benchmark}. Pass --topk_ratio and --alpha explicitly."
        )

    topk_ratio = args.topk_ratio if args.topk_ratio is not None else defaults["topk_ratio"]
    alpha = args.alpha if args.alpha is not None else defaults["alpha"]

    if args.topk_ratio is None or args.alpha is None:
        print(
            f"Using defaults for {args.benchmark}: "
            f"topk_ratio={topk_ratio}, alpha={alpha}"
        )

    return topk_ratio, alpha, [args.layer_start, args.layer_end]


def default_cache_path(args):
    return os.path.join("cache", f"{args.model}_{args.benchmark}_answers.json")


def load_or_generate_cache(args, lvlm, benchmark):
    cache_path = args.cache_path or default_cache_path(args)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f), cache_path

    print(f"No cache found. Generating answers -> {cache_path}")
    cache = {}
    size = benchmark.obtain_size()
    if args.max_samples is not None:
        size = min(size, args.max_samples)

    for idx in tqdm(range(size), desc="Generating answers"):
        sample = benchmark.retrieve(idx)
        if sample is None:
            continue
        answer, generated_ids = lvlm.generate_with_ids(
            sample["img"], sample["question"], args.inference_temp
        )
        cache[str(idx)] = {
            "question": sample["question"],
            "gt_ans": sample["gt_ans"],
            "ans": answer,
            "generated_ids": generated_ids.tolist(),
        }

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Saved answer cache to {cache_path}")
    print(
        "Add a boolean 'label' field to each entry (true=correct, false=incorrect) "
        "using your own evaluator or API, then re-run without --generate_only."
    )
    return cache, cache_path


def tensorize_generated_ids(generated_ids_list, device):
    if isinstance(generated_ids_list, list):
        if generated_ids_list and isinstance(generated_ids_list[0], list):
            return torch.tensor(generated_ids_list, device=device)
        return torch.tensor([generated_ids_list], device=device)
    raise ValueError("generated_ids must be a list")


def evaluate(args):
    fix_seed(args.seed)

    print(f"Loading {args.model} ...")
    lvlm = LLaVA(version=args.model)

    benchmark_cls = BENCHMARK_MAP[args.benchmark]
    benchmark = benchmark_cls()

    cache, cache_path = load_or_generate_cache(args, lvlm, benchmark)

    if args.generate_only:
        print(f"Generation complete. Label entries in {cache_path}, then re-run.")
        return

    topk_ratio, alpha, layer_range = resolve_scoring_params(args)

    size = benchmark.obtain_size()
    if args.max_samples is not None:
        size = min(size, args.max_samples)

    log_dict = {
        "model": args.model,
        "benchmark": args.benchmark,
        "topk_ratio": topk_ratio,
        "alpha": alpha,
        "layer_range": layer_range,
        "cache_path": cache_path,
    }

    y_true, vauq_scores, entropy_scores, is_scores = [], [], [], []

    for idx in tqdm(range(size), desc="Computing VAUQ"):
        idx_str = str(idx)
        if idx_str not in cache:
            continue

        entry = cache[idx_str]
        sample = benchmark.retrieve(idx)
        if sample is None or sample["img"] is None:
            continue

        generated_ids = tensorize_generated_ids(entry["generated_ids"], lvlm.device)
        scores = compute_vauq_scores(
            lvlm,
            sample["img"],
            sample["question"],
            generated_ids,
            topk_ratio=topk_ratio,
            alpha=alpha,
            layer_range=tuple(layer_range),
        )

        log_dict[idx] = {
            "question": sample["question"],
            "gt_ans": sample["gt_ans"],
            "ans": entry["ans"],
            "label": entry.get("label"),
            **scores,
        }

        if entry.get("label") is not None:
            y_true.append(int(entry["label"]))
            vauq_scores.append(-scores["vauq"])
            entropy_scores.append(-scores["entropy"])
            is_scores.append(scores["is_score"])

    log_dict["num_scored"] = len([k for k in log_dict if k.isdigit()])
    log_dict["num_labeled"] = len(y_true)

    metrics = {}
    if len(y_true) == 0:
        print(
            "\nNo labels found in cache. VAUQ scores were computed, but AUROC/AUPR "
            "require a boolean 'label' on each cache entry."
        )
    else:
        y_true = np.array(y_true)
        for name, preds in [
            ("vauq", vauq_scores),
            ("entropy", entropy_scores),
            ("is_score", is_scores),
        ]:
            preds = np.array(preds)
            if len(set(y_true)) < 2:
                metrics[name] = {"auroc": float("nan"), "aupr": float("nan")}
                continue
            auroc = roc_auc_score(y_true, preds)
            aupr = average_precision_score(y_true, preds if auroc >= 0.5 else -preds)
            metrics[name] = {"auroc": float(auroc), "aupr": float(aupr)}

        log_dict["accuracy"] = float(np.mean(y_true))
        log_dict["metrics"] = metrics

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = get_cur_time()
    out_file = os.path.join(
        args.output_dir,
        f"{args.model}_{args.benchmark}_{timestamp}.json",
    )
    with open(out_file, "w") as f:
        json.dump(copy.deepcopy(log_dict), f, indent=2)

    print("\n===== Results =====")
    print(f"Benchmark: {args.benchmark}")
    print(f"Scored samples: {log_dict['num_scored']}")
    print(f"Labeled samples: {log_dict['num_labeled']}")
    print(f"Hyperparameters: topk_ratio={topk_ratio}, alpha={alpha}, layers={layer_range}")
    if metrics:
        print(f"Accuracy: {log_dict['accuracy'] * 100:.2f}%")
        for name, vals in metrics.items():
            print(f"  {name}: AUROC={vals['auroc']:.4f}, AUPR={vals['aupr']:.4f}")
    print(f"\nFull log saved to {out_file}")


if __name__ == "__main__":
    evaluate(parse_args())
