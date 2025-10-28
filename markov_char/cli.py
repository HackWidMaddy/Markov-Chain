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
    """Benchmark Markov model vs brute-force permutation search and report metrics."""
    import random
    from itertools import permutations
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

    n_tokens = len(test) - model.order_k
    if n_tokens <= 0:
        print("Error: insufficient test data")
        return

    alphabet = model.alphabet
    V = len(alphabet)

    # Problem: Given 6 characters, guess the word using brute-force (6! permutations) vs Markov next-char prediction
    target_word_len = args.word_len if args.word_len > 0 else 6
    chars_to_try = args.chars_to_try if args.chars_to_try else None

    if chars_to_try is None:
        # Use a random word from test data (with order_k prefix context)
        if len(test) < model.order_k + target_word_len:
            print(f"Error: need at least {model.order_k + target_word_len} chars in test data")
            return
        start_idx = random.randint(0, len(test) - (model.order_k + target_word_len))
        context = test[start_idx:start_idx + model.order_k]
        target_word = test[start_idx + model.order_k:start_idx + model.order_k + target_word_len]
        available_chars = sorted(list(set(target_word)))  # Unique chars from target word
    else:
        available_chars = sorted(list(set(chars_to_try)))
        # Use a default context
        context = test[:model.order_k] if len(test) >= model.order_k else available_chars[:model.order_k]
        target_word = None  # Will be reconstructed by brute-force

    # Brute-force: enumerate all permutations (6! for 6 chars) to find the word
    brute_force_space_size = math.factorial(len(available_chars)) if len(available_chars) <= 12 else 10**15
    print(f"\nProblem setup:")
    print(f"  Target word length: {target_word_len}")
    print(f"  Available characters: {''.join(available_chars)}")
    print(f"  Brute-force space size: {brute_force_space_size:,} permutations ({len(available_chars)}!)")
    if target_word:
        print(f"  Target word: '{target_word}' (hidden)")

    # Benchmark Markov: backtracking search with iterative next-char prediction
    t0 = time.perf_counter()
    
    def markov_search(current_word, available_chars, history, max_depth):
        """DFS search using Markov predictions, returns (word, num_predictions, found)"""
        if len(current_word) == max_depth:
            return (current_word, 0, True)
        if not available_chars:
            return (current_word, 0, False)
        
        # Get Markov prediction distribution
        dist = model.next_distribution(history)
        
        # Sort available chars by Markov probability (descending)
        char_scores = [(ch, dist.get(ch, 0.0)) for ch in available_chars]
        char_scores.sort(key=lambda x: x[1], reverse=True)
        
        num_predictions = 1  # One call to next_distribution
        
        for char, prob in char_scores:
            remaining = available_chars.replace(char, '', 1)
            new_history = (history + char)[-model.order_k:]
            
            result_word, result_preds, found = markov_search(
                current_word + char, remaining, new_history, max_depth
            )
            num_predictions += result_preds
            
            if found:
                return (result_word, num_predictions, True)
        
        return (current_word, num_predictions, False)
    
    markov_word, markov_steps, markov_success = markov_search("", ''.join(available_chars), context, target_word_len)
    t_markov = time.perf_counter() - t0

    # Benchmark brute-force: enumerate all permutations (6! etc.)
    t0 = time.perf_counter()
    brute_force_steps = 0
    brute_force_found = None
    if target_word and len(available_chars) <= 10:  # Only if reasonable to enumerate
        from itertools import permutations
        for perm in permutations(available_chars, target_word_len):
            brute_force_steps += 1
            candidate = ''.join(perm)
            if candidate == target_word:
                brute_force_found = candidate
                break
            if brute_force_steps > 100000:  # Safety limit
                break
    else:
        brute_force_steps = brute_force_space_size  # Theoretical max
        brute_force_found = "N/A (too large)"
    t_bf = time.perf_counter() - t0

    # Calculate ratio: steps required by brute-force / steps required by Markov
    if markov_steps > 0:
        step_ratio = brute_force_steps / markov_steps
    else:
        step_ratio = brute_force_steps

    # Speedup ratio
    speedup = t_bf / t_markov if t_markov > 0 else 0.0

    # Output
    print("=" * 90)
    print("BENCHMARK: Markov Next-Character Prediction vs Brute-Force Permutation Search")
    print("=" * 90)
    if target_word:
        print(f"\nResults:")
        print(f"  Markov found: '{markov_word}' | Success: {markov_success}")
        print(f"  Brute-force found: '{brute_force_found}'")
        print(f"  Target was: '{target_word}'")
    else:
        print(f"\nResults:")
        print(f"  Markov found: '{markov_word}' | Success: {markov_success}")
        print(f"  Brute-force found: '{brute_force_found}'")
    print(f"\n{'Metric':<30} {'Markov':>20} {'Brute-Force':>20} {'Ratio':>15}")
    print("-" * 90)
    print(f"{'Steps to solution':<30} {markov_steps:>20} {brute_force_steps:>20,} {step_ratio:>14.2f}x")
    print(f"{'Search space size':<30} {'O(order_k)':>20} {brute_force_space_size:>20,}")
    print(f"{'Time taken (ms)':<30} {t_markov*1000:>20.2f} {t_bf*1000:>20.2f} {speedup:>14.2f}x")
    print(f"{'Success rate':<30} {'✓' if markov_success else '✗':>20} {'✓' if brute_force_found else '✗':>20}")
    print("-" * 90)
    step_reduction = ((brute_force_steps - markov_steps) / max(1, brute_force_steps)) * 100
    print(f"\nKey Insight:")
    print(f"  • Brute-force needs {brute_force_steps:,} steps (enumerating all permutations = {len(available_chars)}!)")
    print(f"  • Markov needs {markov_steps} prediction function calls (with backtracking)")
    print(f"  • Markov tries candidates in probability order, reducing search space")
    print(f"  • Step reduction: {step_reduction:.1f}% (Markov makes {step_ratio:.1f}x fewer calls than brute-force)")
    print(f"  • Complexity: Markov O({len(available_chars)}) predictions vs brute-force O({len(available_chars)}!) tries")
    print("=" * 90)


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

    pb = sub.add_parser("benchmark", help="Compare Markov vs brute-force permutation search")
    pb.add_argument("--model", required=True)
    pb.add_argument("--input", required=True)
    pb.add_argument("--train-frac", type=float, default=0.8)
    pb.add_argument("--val-frac", type=float, default=0.1)
    pb.add_argument("--seed", type=int, default=42)
    pb.add_argument("--seed-bench", type=int, default=42, help="Random seed for benchmark")
    pb.add_argument("--n-tokens", type=int, default=None, help="Limit test tokens")
    pb.add_argument("--word-len", type=int, default=6, help="Length of word to guess")
    pb.add_argument("--chars-to-try", type=str, default=None, help="Specific characters to use (e.g., 'abcdef')")
    pb.set_defaults(func=cmd_benchmark)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


