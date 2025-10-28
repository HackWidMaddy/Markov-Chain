## Character-level Markov Chain (k-order) for Next-Character Prediction

### Quickstart

1. Prepare a plain text file, e.g., `data.txt`.
2. Train and evaluate (Windows PowerShell example):
```bash
python -m markov_char.cli train --input data.txt --order-k 2 --add-alpha 0.5 --lowercase --model-out model.json
```
3. Evaluate a saved model:
```bash
python -m markov_char.cli eval --model model.json --input data.txt
```
4. Generate text:
```bash
python -m markov_char.cli generate --model model.json --seed-text "the " --length 300 --temperature 0.9
```

### What it does
- Trains a k-order character Markov model with optional add-α smoothing.
- Reports accuracy, cross-entropy, and perplexity on val/test splits.
- Saves/loads model to JSON.
- Generates text with temperature-controlled sampling.

### Notable flags
- `--order-k` (int): Markov order (memory length).
- `--add-alpha` (float): Additive smoothing coefficient (0 = no smoothing).
- `--lowercase`: Normalize to lowercase.
- `--strip-whitespace`: Remove spaces/tabs/newlines.
- `--remove-punctuation`: Remove punctuation.
- `--train-frac`, `--val-frac`, `--seed`: Control splits.

### Requirements
- Python 3.9+
- Optional: `matplotlib` (for future plotting extension)

### Notes
- Higher `k` needs more data due to state space growth.
- Temperature < 1.0 makes outputs more deterministic; > 1.0 increases randomness.
