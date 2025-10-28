from __future__ import annotations

import os
import random
import string
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class TextPrepConfig:
    lowercase: bool = False
    keep_whitespace: bool = True
    keep_punctuation: bool = True
    allowed_extra_chars: str = ""  # e.g., "\n\t"


def load_text(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def normalize_text(raw: str, cfg: TextPrepConfig) -> str:
    text = raw.lower() if cfg.lowercase else raw
    allowed = set(cfg.allowed_extra_chars)
    allowed.update(string.ascii_lowercase if cfg.lowercase else string.ascii_letters)
    if cfg.keep_whitespace:
        allowed.update(" \t\r\n")
    if cfg.keep_punctuation:
        allowed.update(string.punctuation)
    allowed.update(string.digits)
    return "".join(ch for ch in text if ch in allowed)


def build_alphabet(text: str) -> List[str]:
    return sorted(list({ch for ch in text}))


def split_text(text: str, train_frac: float = 0.8, val_frac: float = 0.1, seed: int = 42) -> Tuple[str, str, str]:
    if not (0.0 < train_frac < 1.0) or not (0.0 <= val_frac < 1.0) or train_frac + val_frac < 0.0 or train_frac + val_frac >= 1.0:
        raise ValueError("Invalid split fractions")
    random.seed(seed)
    n = len(text)
    # Simple contiguous split to preserve local dependencies
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    train = text[:train_end]
    val = text[train_end:val_end]
    test = text[val_end:]
    return train, val, test


