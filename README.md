# SNN Formal Language Learning

Compare GRU (RNN), LSTM, a feedforward Spiking Neural Network (SNN), and a recurrent spiking control (RSNN) as recognizers of formal languages — following the setup in *Training Neural Networks as Recognizers of Formal Languages*.

## Goal

Investigate whether SNNs (which process symbols as temporal spikes) can learn formal languages and generalize to longer sequences, compared to classical recurrent baselines.

## Quick Start

```bash
pip install -r requirements.txt
python -m formal_language_snn --config configs/config.yaml
```

Override config values:

```bash
python -m formal_language_snn --set experiment.language=palindrome --set training.epochs=10
```

Structured experiments (10 seeds by default; use `--pilot` for a single-seed smoke test):

```bash
python scripts/exp1_length.py --plot
python scripts/exp2_beta_sweep.py --plot
python scripts/exp3_difficulty.py --plot
```

Before any paper run, the symmetry guard must pass — it enforces that only the factor under study varies across models, languages and runs:

```bash
python scripts/check_symmetry.py
```

## Documentation

- **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** — comprehensive guide to languages, negative strategies, models, data pipeline, and all three structured experiments
- **[EXPERIMENT_COMMANDS.md](EXPERIMENT_COMMANDS.md)** — command reference and config cheat sheet


## Utilities

- [`scripts/combine_exp3_figs.py`](scripts/combine_exp3_figs.py) — stitch the Exp3 panels into the combined paper figure
- [`printAllFiles.sh`](printAllFiles.sh) — dump repo source to a text file (output is gitignored; it goes stale the moment the code changes, so regenerate rather than reading an old dump)
