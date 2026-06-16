import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy


class OutputScoreInfo:
    """Token-level uncertainty metrics from model logits."""

    def __init__(self, response_logits, response_ids, device):
        self.response_logits = response_logits
        self.response_ids = response_ids
        self.device = device
        self.probs = F.softmax(
            self.response_logits.to(device=self.device, dtype=torch.float32),
            dim=-1,
        )
        self.all_token_probs_list = self.probs.cpu().tolist()

    def compute_entropy(self):
        seq_entropy_list = [
            entropy(np.array(p_dist, dtype=np.float32), base=2)
            for p_dist in self.all_token_probs_list
        ]
        return float(np.mean(seq_entropy_list))
