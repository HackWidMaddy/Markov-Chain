from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from .data import TextPrepConfig, build_alphabet, load_text, normalize_text, split_text
from .metrics import cross_entropy_and_accuracy, perplexity
from .model import CharMarkovModel, MarkovConfig


def cmd_train(args: argparse.Namespace) -> None:
    raw = load_text(args.input)
    prep_cfg = TextPrepConfig(
        lowercase=args.lowercase,
        keep_whitespace=not args.strip_whitespace,
        keep_punctuation=not args.remove_punctuation,
        allowed_extra_chars=args.allowed_extra or "",
    )
    text = normalize_text(raw, prep_cfg)
    train, val, test = split_text(text, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed)

    cfg = MarkovConfig(order_k=args.order_k, add_alpha=args.add_alpha, lowercase=args.lowercase, use_witten_bell=args.witten_bell)
    model = CharMarkovModel(cfg)
    model.fit(train)

    val_ce, val_acc, n_val = cross_entropy_and_accuracy(model, val)
    test_ce, test_acc, n_test = cross_entropy_and_accuracy(model, test)

    print(f"Validation: n={n_val} acc={val_acc:.4f} xent={val_ce:.4f} ppl={perplexity(val_ce):.2f}")
    print(f"Test:       n={n_test} acc={test_acc:.4f} xent={test_ce:.4f} ppl={perplexity(test_ce):.2f}")

    if args.model_out:
        model.save(args.model_out)
        print(f"Saved model to {args.model_out}")


def cmd_eval(args: argparse.Namespace) -> None:
    model = CharMarkovModel.load(args.model)
    raw = load_text(args.input)
    prep_cfg = TextPrepConfig(
        lowercase=model.lowercase,
        keep_whitespace=True,
        keep_punctuation=True,
        allowed_extra_chars="\n\t\r",
    )
    text = normalize_text(raw, prep_cfg)
    _, _, test = split_text(text, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed)
    ce, acc, n = cross_entropy_and_accuracy(model, test)
    print(json.dumps({
        "n": n,
        "accuracy": acc,
        "cross_entropy": ce,
        "perplexity": perplexity(ce),
    }, indent=2))


def cmd_generate(args: argparse.Namespace) -> None:
    model = CharMarkovModel.load(args.model)
    seed = args.seed_text or ""  # may be empty
    if args.seed_rng is not None:
        import random
        random.seed(args.seed_rng)
    out = model.generate(seed=seed, length=args.length, temperature=args.temperature)
    print(out)


def cmd_probs(args: argparse.Namespace) -> None:
    model = CharMarkovModel.load(args.model)
    dist = model.next_distribution(args.context)
    items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    topk = args.topk if args.topk is not None else len(items)
    for ch, p in items[:topk]:
        safe = ch
        if ch == "\n":
            safe = "\\n"
        elif ch == "\t":
            safe = "\\t"
        elif ch == " ":
            safe = "␠"  # visualize space
        print(f"{safe}\t{p:.6f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Character-level Markov chain for next-char prediction")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train model from text file and evaluate")
    pt.add_argument("--input", required=True, help="Path to input text file")
    pt.add_argument("--order-k", type=int, default=1)
    pt.add_argument("--add-alpha", type=float, default=0.0)
    pt.add_argument("--witten-bell", action="store_true", help="Use Witten–Bell interpolated backoff across 1..k")
    pt.add_argument("--lowercase", action="store_true")
    pt.add_argument("--strip-whitespace", action="store_true")
    pt.add_argument("--remove-punctuation", action="store_true")
    pt.add_argument("--allowed-extra", default="\n\t\r")
    pt.add_argument("--train-frac", type=float, default=0.8)
    pt.add_argument("--val-frac", type=float, default=0.1)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--model-out", default=None)
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("eval", help="Evaluate a saved model on a file")
    pe.add_argument("--model", required=True)
    pe.add_argument("--input", required=True)
    pe.add_argument("--train-frac", type=float, default=0.8)
    pe.add_argument("--val-frac", type=float, default=0.1)
    pe.add_argument("--seed", type=int, default=42)
    pe.set_defaults(func=cmd_eval)

    pg = sub.add_parser("generate", help="Generate text from a saved model")
    pg.add_argument("--model", required=True)
    pg.add_argument("--seed-text", default="")
    pg.add_argument("--length", type=int, default=200)
    pg.add_argument("--temperature", type=float, default=1.0)
    pg.add_argument("--seed-rng", type=int, default=None, help="Random seed for sampling")
    pg.set_defaults(func=cmd_generate)

    pp = sub.add_parser("probs", help="Show next-char probabilities for a context")
    pp.add_argument("--model", required=True)
    pp.add_argument("--context", required=True, help="History/context string")
    pp.add_argument("--topk", type=int, default=20)
    pp.set_defaults(func=cmd_probs)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


