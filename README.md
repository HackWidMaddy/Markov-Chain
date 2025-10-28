## Character-level Markov Chain (k-order) for Next-Character Prediction

This project trains a character-level Markov model to predict the next character given the last k characters (order k). It supports classic add-α (Laplace) smoothing and Witten–Bell interpolated backoff for better accuracy on unseen contexts.

### How it works (short)
- Training scans the text and counts transitions from each context (last k chars) to the next char.
- Smoothing avoids zero-probabilities. Add-α adds a small mass to every symbol; Witten–Bell backs off to lower orders based on how many unique next symbols were seen.
- Prediction uses either the precomputed k-gram probabilities (add-α) or Witten–Bell interpolation across orders 1..k.
- Generation samples or picks the argmax given a seed string.

### Quickstart (Windows PowerShell)

1) Prepare a plain text file, e.g., `markov_char\data.txt`.

2) Train and evaluate (order 2, add-α smoothing):
```powershell
python -m markov_char.cli train --input markov_char\data.txt --order-k 2 --add-alpha 0.5 --lowercase --model-out model.json
```

3) Train with higher accuracy using Witten–Bell interpolation (recommended when you have enough data):
```powershell
python -m markov_char.cli train --input markov_char\data.txt --order-k 3 --witten-bell --lowercase --model-out model.json
```

4) Evaluate a saved model:
```powershell
python -m markov_char.cli eval --model model.json --input markov_char\data.txt
```

5) Generate text:
```powershell
# Deterministic (argmax)
python -m markov_char.cli generate --model model.json --seed-text "interconnec" --length 1 --temperature 0

# Stochastic with fixed RNG seed (reproducible)
python -m markov_char.cli generate --model model.json --seed-text "interconnec" --length 20 --temperature 0.9 --seed-rng 123
```

6) Inspect next-char probabilities for a context:
```powershell
python -m markov_char.cli probs --model model.json --context "interconnec" --topk 10
```

7) Benchmark against brute-force baseline (accuracy, perplexity, search-space reduction):
```powershell
python -m markov_char.cli benchmark --model model.json --input markov_char\data.txt
```

### Commands and flags
- `train`:
  - `--input`: path to text file
  - `--order-k`: Markov order (memory length). Larger k needs more data.
  - `--add-alpha`: add-α smoothing (0 disables). Used when NOT using Witten–Bell.
  - `--witten-bell`: enable Witten–Bell interpolated backoff across orders 1..k
  - `--lowercase`: lowercase normalization at training time
  - `--strip-whitespace`, `--remove-punctuation`, `--allowed-extra`: control preprocessing
  - `--train-frac`, `--val-frac`, `--seed`: dataset split and reproducibility
  - `--model-out`: save trained model (JSON)

- `eval`:
  - `--model`: path to model JSON
  - `--input`: path to text (same domain/preprocessing as training)
  - `--train-frac`, `--val-frac`, `--seed`: to derive test split from the file

- `generate`:
  - `--model`: path to model JSON
  - `--seed-text`: initial text; only last k chars are used as context
  - `--length`: number of characters to generate
  - `--temperature`: 0 = argmax (deterministic), >0 applies stochastic sampling
  - `--seed-rng`: random seed for reproducible sampling

- `probs`:
  - `--model`: path to model JSON
  - `--context`: show next-char distribution for this history
  - `--topk`: number of top characters to print

- `benchmark`:
  - `--model`: path to model JSON
  - `--input`: path to text (same domain/preprocessing as training)
  - `--train-frac`, `--val-frac`, `--seed`: to derive test split from the file
  - `--n-tokens`: limit number of test tokens (optional)
  - `--seed-bench`: random seed for brute-force baseline (default: 42)

### Smoothing methods
- Add-α (Laplace): adds α count to each symbol before normalizing. Good baseline; too large α can over-flatten probabilities.
- Witten–Bell backoff (recommended): interpolates higher-order estimates with lower orders according to the ratio of total events (N) vs unique continuations (T) in the context:
  - λ = N / (N + T)
  - P_k = λ · MLE_k + (1 − λ) · P_{k−1}
  - Falls back to uniform at order 0. Helps greatly with unseen/rare contexts.

### Troubleshooting
- "ModuleNotFoundError: No module named 'markov_char'": run commands from the project root (e.g., `D:\CNS ISE`), not inside the `markov_char` folder. Or set `PYTHONPATH` to the parent directory.
- Predictions look random for short contexts: ensure your prompt matches training normalization (e.g., use lowercase if you trained with `--lowercase`). With `temperature 0`, you’ll see the deterministic argmax.
- Very flat probabilities: increase data size, reduce `add-alpha`, or enable `--witten-bell`. Consider increasing `--order-k` if you have enough data.

### Maximizing accuracy
- Data size and domain match matter most. Use more text similar to your prompts.
- Choose order k appropriate to data: k=2–3 for small datasets; k=3–5 for larger.
- Prefer `--witten-bell` for better generalization to unseen contexts.
- Normalize consistently: if you train with `--lowercase`, prompt in lowercase.
- Evaluate via perplexity/accuracy on a held-out test split and pick k that minimizes perplexity.
- For hard accuracy comparisons, use `--temperature 0` when generating.

### Benchmark output

The `benchmark` command compares your trained model against a brute-force (uniform random) baseline and reports:

- **Accuracy**: percentage of correct next-character predictions
- **Cross-entropy**: average bits per prediction (lower is better)
- **Perplexity**: effective uncertainty in next-char distribution (lower is better)
- **Search space reduction**: how much the model narrows the effective alphabet size
- **Speed**: prediction rate in tokens/second
- **Improvement**: percentage gains over brute-force

Example output shows the Markov model typically achieves 20–50% accuracy vs ~3% for uniform, with 90%+ search-space reduction.

### Requirements
- Python 3.9+
- Optional: `matplotlib` (future plotting extension)
