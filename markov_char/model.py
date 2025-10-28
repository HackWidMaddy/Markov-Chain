from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import os
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


State = Tuple[str, ...]


def _normalize_distribution(counts: Mapping[str, float]) -> Dict[str, float]:
    total = float(sum(counts.values()))
    if total <= 0.0:
        return {k: 0.0 for k in counts.keys()}
    return {k: v / total for k, v in counts.items()}


def _argmax(mapping: Mapping[str, float]) -> Optional[str]:
    best_key: Optional[str] = None
    best_val = -math.inf
    for k, v in mapping.items():
        if v > best_val:
            best_key = k
            best_val = v
    return best_key


@dataclass
class MarkovConfig:
    order_k: int = 1
    add_alpha: float = 0.0
    lowercase: bool = False
    use_witten_bell: bool = False


class CharMarkovModel:
    def __init__(self, config: Optional[MarkovConfig] = None) -> None:
        self.config = config or MarkovConfig()
        self.order_k: int = self.config.order_k
        if self.order_k < 1:
            raise ValueError("order_k must be >= 1")
        self.add_alpha: float = self.config.add_alpha
        if self.add_alpha < 0.0:
            raise ValueError("add_alpha must be >= 0.0")

        self.lowercase: bool = self.config.lowercase
        self.use_witten_bell: bool = self.config.use_witten_bell

        self.alphabet: List[str] = []
        self.alphabet_set: set[str] = set()
        self.state_to_counts: Dict[State, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.state_to_probs: Dict[State, Dict[str, float]] = {}
        # For Witten–Bell, we maintain counts for all orders 1..k
        self.order_to_counts: List[Dict[State, Dict[str, int]]] = []

    def _normalize_text(self, text: str) -> str:
        return text.lower() if self.lowercase else text

    def fit(self, text: str, alphabet: Optional[Sequence[str]] = None) -> None:
        text = self._normalize_text(text)
        if not text:
            raise ValueError("Empty training text")

        if alphabet is None:
            self.alphabet = sorted(list({ch for ch in text}))
        else:
            self.alphabet = list(dict.fromkeys(alphabet))
        self.alphabet_set = set(self.alphabet)
        if not self.alphabet:
            raise ValueError("Alphabet is empty after preprocessing")

        self.state_to_counts.clear()
        self.order_to_counts = [defaultdict(lambda: defaultdict(int)) for _ in range(self.order_k)]

        window: List[str] = []
        for ch in text:
            if len(window) < self.order_k:
                window.append(ch)
                continue
            # Update counts for all orders m=1..k using the same next char
            for m in range(1, self.order_k + 1):
                state_m: State = tuple(window[-m:])
                self.order_to_counts[m - 1][state_m][ch] += 1
            # Back-compat storage for highest order
            state: State = tuple(window[-self.order_k :])
            self.state_to_counts[state][ch] += 1
            window.append(ch)

        self._compute_probabilities()

    def _compute_probabilities(self) -> None:
        self.state_to_probs = {}
        V = len(self.alphabet)
        alpha = self.add_alpha
        if self.use_witten_bell:
            # No precomputed probs; we use counts dynamically
            return
        for state, next_counts in self.state_to_counts.items():
            smoothed: Dict[str, float] = {ch: float(next_counts.get(ch, 0)) for ch in self.alphabet}
            if alpha > 0.0:
                for ch in self.alphabet:
                    smoothed[ch] = smoothed[ch] + alpha
            self.state_to_probs[state] = _normalize_distribution(smoothed)

    def next_distribution(self, history: Sequence[str]) -> Dict[str, float]:
        if not self.state_to_probs:
            # For WB we don't precompute; ensure we at least have alphabet
            if not self.alphabet:
                raise RuntimeError("Model not fitted")
        hist_text = self._normalize_text("".join(history))
        hist_k = tuple(hist_text[-self.order_k :])
        if not self.use_witten_bell:
            if len(hist_k) < self.order_k:
                return {ch: 1.0 / len(self.alphabet) for ch in self.alphabet}
            if hist_k in self.state_to_probs:
                return dict(self.state_to_probs[hist_k])
            if self.add_alpha > 0.0:
                prior = {ch: self.add_alpha for ch in self.alphabet}
                return _normalize_distribution(prior)
            return {ch: 1.0 / len(self.alphabet) for ch in self.alphabet}

        # Witten–Bell interpolated backoff across orders
        def wb_distribution(context: Tuple[str, ...], order: int) -> Dict[str, float]:
            if order <= 0:
                return {ch: 1.0 / len(self.alphabet) for ch in self.alphabet}
            counts_map = self.order_to_counts[order - 1]
            counts = counts_map.get(context, {})
            N = sum(counts.values())
            T = len([c for c in counts.values() if c > 0])
            # Lower-order distribution
            backoff = wb_distribution(context[1:] if len(context) > 0 else context, order - 1)
            if N + T == 0:
                return backoff
            lam = N / (N + T)
            # MLE for this order
            mle = {ch: (counts.get(ch, 0) / N) if N > 0 else 0.0 for ch in self.alphabet}
            # Interpolate
            return {ch: lam * mle[ch] + (1.0 - lam) * backoff[ch] for ch in self.alphabet}

        context = hist_k[-self.order_k :]
        return wb_distribution(context, len(context))

    def predict_next(self, history: Sequence[str]) -> str:
        dist = self.next_distribution(history)
        best = _argmax(dist)
        if best is None:
            raise RuntimeError("Empty distribution")
        return best

    def sample_next(self, history: Sequence[str], temperature: float = 1.0) -> str:
        import random

        dist = self.next_distribution(history)
        if temperature <= 0.0:
            return self.predict_next(history)

        scaled: Dict[str, float] = {}
        for ch, p in dist.items():
            if p <= 0.0:
                scaled[ch] = 0.0
            else:
                scaled[ch] = p ** (1.0 / max(1e-8, temperature))
        scaled = _normalize_distribution(scaled)

        r = random.random()
        acc = 0.0
        last_ch = self.alphabet[-1]
        for ch in self.alphabet:
            acc += scaled.get(ch, 0.0)
            if r <= acc:
                return ch
            last_ch = ch
        return last_ch

    def generate(self, seed: Sequence[str], length: int, temperature: float = 1.0) -> str:
        history: List[str] = list(self._normalize_text("".join(seed)))
        out: List[str] = []
        for _ in range(length):
            ch = self.sample_next(history, temperature=temperature)
            out.append(ch)
            history.append(ch)
        return "".join(out)

    def to_dict(self) -> Dict[str, object]:
        serializable: Dict[str, object] = {
            "order_k": self.order_k,
            "add_alpha": self.add_alpha,
            "lowercase": self.lowercase,
            "use_witten_bell": self.use_witten_bell,
            "alphabet": self.alphabet,
        }
        if not self.use_witten_bell:
            serializable["state_to_probs"] = {
                "|".join(state): probs for state, probs in self.state_to_probs.items()
            }
        else:
            # Serialize counts for WB
            serializable["order_to_counts"] = [
                {"|".join(state): counts for state, counts in od.items()} for od in self.order_to_counts
            ]
        return serializable

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CharMarkovModel":
        cfg = MarkovConfig(
            order_k=int(data.get("order_k", 1)),
            add_alpha=float(data.get("add_alpha", 0.0)),
            lowercase=bool(data.get("lowercase", False)),
            use_witten_bell=bool(data.get("use_witten_bell", False)),
        )
        model = cls(cfg)
        model.alphabet = list(data.get("alphabet", []))  # type: ignore[arg-type]
        model.alphabet_set = set(model.alphabet)
        if model.use_witten_bell:
            model.order_to_counts = []
            raw_list = data.get("order_to_counts", [])  # type: ignore[assignment]
            for od in raw_list:
                decoded: Dict[State, Dict[str, int]] = {}
                for state_key, counts in od.items():
                    decoded[tuple(state_key.split("|"))] = {k: int(v) for k, v in counts.items()}
                model.order_to_counts.append(decoded)
        else:
            raw_probs: Mapping[str, Dict[str, float]] = data.get("state_to_probs", {})  # type: ignore[assignment]
            model.state_to_probs = {}
            for state_key, probs in raw_probs.items():
                state = tuple(state_key.split("|"))
                model.state_to_probs[state] = dict(probs)
        return model

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharMarkovModel":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


