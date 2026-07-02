# SNN Formal Language Learning

Compare GRU (RNN) and Spiking Neural Network (SNN) models as recognizers of formal languages — following the setup in *Training Neural Networks as Recognizers of Formal Languages*.

## Goal

Investigate whether SNNs (which process symbols as temporal spikes) can learn formal languages and generalize to longer sequences, compared to a classical RNN baseline.

## Languages

Three context-free focus languages:

1. **anbn** — a^n b^n
2. **balanced_parens** — balanced parentheses (Dyck language)
3. **palindrome** — even-length palindromes over {a, b, c}

## Quick Start

```bash
pip install -r requirements.txt
python -m formal_language_snn --config configs/config.yaml
```

Override config values:

```bash
python -m formal_language_snn --set experiment.language=palindrome --set training.epochs=10
```

## Experiment Plan

Structured experiments use **10 seeds** by default (mean ± std). Use `--pilot` for a single-seed smoke test.

| Experiment | What it measures |
|------------|------------------|
| **Exp 1 — Length** | Accuracy vs word-length buckets: 1–10, 11–20, 21–40, 41–80, 81–160 |
| **Exp 2 — Beta** | SNN membrane decay β sweep (fixed β, `learn_beta=false`) |
| **Exp 3 — Difficulty** | Hard vs easy negative examples per language |

```bash
python scripts/exp1_length.py --plot
python scripts/exp2_beta_sweep.py --plot
python scripts/exp3_difficulty.py --plot
```

Plot or export aggregated JSON:

```bash
python scripts/plot_experiments.py --experiment exp1 --input outputs/experiments/exp1_length/exp1_length_anbn_seedagg.json
python scripts/aggregate_experiments.py --input outputs/experiments/exp1_length/exp1_length_anbn_seedagg.json
```

Convenience wrapper:

```bash
./scripts/run_experiment.sh default   # single run from configs/config.yaml
./scripts/run_experiment.sh all       # all three languages
```

## Configuration

- [`configs/config.yaml`](configs/config.yaml) — default single-run settings
- [`configs/experiments/`](configs/experiments/) — presets for exp1, exp2, exp3

Results are written as JSON under `outputs/`.

## Models

- **ClassicRNN** — GRU + MLP classifier (final hidden state)
- **SpikingNet** — two-layer LIF network; `SF.ce_rate_loss()` for training, spike sum for inference

## Utilities

- [`printAllFiles.sh`](printAllFiles.sh) — dump repo source to a text file (skips `.venv`, `outputs`, etc.)
