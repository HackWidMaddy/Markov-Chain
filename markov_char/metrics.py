from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

from .model import CharMarkovModel


def cross_entropy_and_accuracy(model: CharMarkovModel, text: str) -> Tuple[float, float, int]:
    if len(text) <= model.order_k:
        return float("nan"), float("nan"), 0
    log_loss_sum = 0.0
    num_tokens = 0
    correct = 0
    history: Sequence[str]
    for i in range(model.order_k, len(text)):
        history = text[i - model.order_k : i]
        true_ch = text[i]
        dist: Dict[str, float] = model.next_distribution(history)
        p = max(1e-12, dist.get(true_ch, 0.0))
        log_loss_sum += -math.log(p)
        num_tokens += 1
        pred = model.predict_next(history)
        if pred == true_ch:
            correct += 1
    if num_tokens == 0:
        return float("nan"), float("nan"), 0
    cross_ent = log_loss_sum / num_tokens
    acc = correct / num_tokens
    return cross_ent, acc, num_tokens


def perplexity(cross_entropy_nats: float) -> float:
    if not math.isfinite(cross_entropy_nats):
        return float("nan")
    return math.exp(cross_entropy_nats)


