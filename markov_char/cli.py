from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, Optional, Tuple

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


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark Markov model vs brute-force (uniform baseline) and report metrics."""
    import random
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

    # Prune test to requested n if specified
    if args.n_tokens is not None and args.n_tokens < len(test):
        test = test[:args.n_tokens]

    n_tokens = len(test) - model.order_k
    if n_tokens <= 0:
        print("Error: insufficient test data")
        return

    # Build brute-force baseline: uniform probabilities
    alphabet = model.alphabet
    V = len(alphabet)

    # Benchmark Markov
    t0 = time.perf_counter()
    markov_correct = 0
    markov_ce_sum = 0.0
    markov_dist_count = 0
    for i in range(model.order_k, len(test)):
        history = test[i - model.order_k : i]
        true_ch = test[i]
        dist = model.next_distribution(history)
        p_true = max(1e-12, dist.get(true_ch, 0.0))
        markov_ce_sum += -math.log2(p_true) if p_true > 1e-12 else 20.0
        markov_dist_count += 1
        pred = model.predict_next(history)
        if pred == true_ch:
            markov_correct += 1
    t_markov = time.perf_counter() - t0

    # Benchmark brute-force (uniform sampling)
    random.seed(args.seed_bench)
    t0 = time.perf_counter()
    bf_correct = 0
    bf_ce_sum = 0.0
    uniform_prob = 1.0 / V
    uniform_ce = -math.log2(uniform_prob)
    for i in range(model.order_k, len(test)):
        history = test[i - model.order_k : i]
        true_ch = test[i]
        # Always uniform; no context
        p_true = uniform_prob
        bf_ce_sum += uniform_ce
        pred = random.choice(alphabet)
        if pred == true_ch:
            bf_correct += 1
    t_bf = time.perf_counter() - t0

    # Metrics
    markov_acc = markov_correct / n_tokens if n_tokens > 0 else 0.0
    bf_acc = bf_correct / n_tokens if n_tokens > 0 else 0.0
    markov_ce = markov_ce_sum / markov_dist_count if markov_dist_count > 0 else float("inf")
    bf_ce = bf_ce_sum / n_tokens if n_tokens > 0 else float("inf")
    markov_ppl = perplexity(markov_ce)
    bf_ppl = perplexity(bf_ce)

    # Search space reduction: effective alphabet size from entropy
    markov_eff_v = 2 ** (markov_ce / 1.442695)
    bf_eff_v = V
    reduction = 1.0 - (markov_eff_v / bf_eff_v) if bf_eff_v > 0 else 0.0

    # Speedup
    speedup = t_bf / t_markov if t_markov > 0 else 0.0

    # Output
    print("=" * 70)
    print("BENCHMARK: Markov Chain vs Brute-Force (Uniform Baseline)")
    print("=" * 70)
    print(f"\nTest configuration:")
    print(f"  Tokens: {n_tokens}")
    print(f"  Alphabet size: {V}")
    print(f"  Model order: {model.order_k}")
    print(f"  Using Witten-Bell: {model.use_witten_bell}")
    print(f"\n{'Metric':<25} {'Markov':>15} {'Brute-Force':>15} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {markov_acc:>15.4%} {bf_acc:>15.4%} {(markov_acc / bf_acc - 1.0) * 100 if bf_acc > 0 else 0:>14.1f}%")
    print(f"{'Cross-Entropy (bits)':<25} {markov_ce:>15.4f} {bf_ce:>15.4f} {((bf_ce / markov_ce - 1.0) * 100) if markov_ce > 0 else 0:>14.1f}%")
    print(f"{'Perplexity':<25} {markov_ppl:>15.2f} {bf_ppl:>15.2f} {((bf_ppl / markov_ppl - 1.0) * 100) if markov_ppl > 0 else 0:>14.1f}%")
    print(f"{'Prediction time (s)':<25} {t_markov:>15.6f} {t_bf:>15.6f} {speedup:>14.2f}x")
    print(f"{'Pred rate (tok/s)':<25} {n_tokens / t_markov if t_markov > 0 else 0:>15.0f} {n_tokens / t_bf if t_bf > 0 else 0:>15.0f} {speedup:>14.2f}x")
    print("-" * 70)
    print(f"\nSearch space reduction:")
    print(f"  Effective alphabet (Markov): {markov_eff_v:.2f}")
    print(f"  Effective alphabet (Uniform): {bf_eff_v:.2f}")
    print(f"  Reduction: {reduction:.1%}")
    print(f"\nInterpretation:")
    print(f"  • Markov model reduces effective search space by {reduction:.1%}")
    print(f"  • Accuracy improves by {(markov_acc / bf_acc - 1.0) * 100 if bf_acc > 0 else 0:.1f}% vs uniform baseline")
    print(f"  • Perplexity {(bf_ppl / markov_ppl - 1.0) * 100 if markov_ppl > 0 else 0:.1f}% lower (lower is better)")
    print(f"  • Prediction time: {t_markov*1000:.2f}ms total for {n_tokens} tokens")
    print("=" * 70)


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

    pb = sub.add_parser("benchmark", help="Compare Markov vs brute-force baseline")
    pb.add_argument("--model", required=True)
    pb.add_argument("--input", required=True)
    pb.add_argument("--train-frac", type=float, default=0.8)
    pb.add_argument("--val-frac", type=float, default=0.1)
    pb.add_argument("--seed", type=int, default=42)
    pb.add_argument("--seed-bench", type=int, default=42, help="Random seed for brute-force")
    pb.add_argument("--n-tokens", type=int, default=None, help="Limit test tokens")
    pb.set_defaults(func=cmd_benchmark)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


