"""VAUQ scoring: predictive entropy + Image-Information Score (IS)."""

import torch

from metrics import OutputScoreInfo


def compute_entropy(lvlm, image, question, generated_ids):
    with torch.no_grad():
        logits = lvlm.get_logits(image, question, generated_ids)
        score_info = OutputScoreInfo(logits, generated_ids, lvlm.device)
        return score_info.compute_entropy()


def compute_entropy_core_masked(
    lvlm, image, question, generated_ids, topk_ratio, layer_range
):
    with torch.no_grad():
        logits = lvlm.get_logits_masked(
            image,
            question,
            generated_ids,
            topk_ratio=topk_ratio,
            layer_range=tuple(layer_range),
        )
        score_info = OutputScoreInfo(logits, generated_ids, lvlm.device)
        return score_info.compute_entropy()


def compute_vauq_scores(
    lvlm,
    image,
    question,
    generated_ids,
    topk_ratio=0.6,
    alpha=0.5,
    layer_range=(10, 25),
):
    """
    Compute VAUQ uncertainty scores for one sample.

    Returns:
        entropy_org: predictive entropy H(Y|X,V)
        is_score: Image-Information Score (PMI under core masking)
        vauq: combined score = alpha * IS - entropy (higher => more likely correct)
    """
    entropy_org = compute_entropy(lvlm, image, question, generated_ids)
    entropy_masked = compute_entropy_core_masked(
        lvlm, image, question, generated_ids, topk_ratio, layer_range
    )

    is_score = entropy_masked - entropy_org
    vauq = alpha * is_score - entropy_org

    return {
        "entropy": entropy_org,
        "is_score": is_score,
        "entropy_masked": entropy_masked,
        "vauq": vauq,
    }
