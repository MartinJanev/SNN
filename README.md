# SNN Formal Language Learning

An experimental framework comparing RNN and Spiking Neural Network (SNN) models on learning formal languages.

## Goal
To investigate whether spiking neural networks (which process symbols as temporal spikes) can learn formal languages as effectively as classic RNNs, and to compare their abilities to generalize to longer, unseen sequences.

## Quick Start
```bash
pip install -r requirements.txt
python main.py
```

## Languages Supported
1. **anbn**: a^n b^n (context-free)
2. **anbncn**: a^n b^n c^n (context-sensitive)
3. **palindrome**: Even-length palindromes
4. **paren**: Balanced parentheses
5. **equal_ab**: Equal counts of 'a' and 'b'
6. **ends_with_abb**: Strings ending with 'abb'
7. **repeat_ab**: (ab)^n repetition
8. **ww**: w concatenated with w (non-context-free)
9. **prime_a**: a^p where p is prime
10. **alternating**: No two consecutive identical symbols

## Configuration System
Instead of passing many arguments, experiment parameters are loaded from YAML files. You can find these in the `configs/` directory:
- `configs/default.yaml` - Default parameters
- `configs/quick.yaml` - Fast testing
- `configs/comprehensive.yaml` - Harder evaluation bounds
- `configs/anbncn.yaml` - Context-sensitive language defaults

You can override any setting using CLI flags:
```bash
python main.py --language palindrome --epochs 10 --train-pairs 500
python main.py --config configs/quick.yaml --language paren
```

## Convenience Scripts
Easily run common experiments:
```bash
# Windows
run_experiment.bat quick
run_experiment.bat all-quick   # Tests all languages

# MacOS / Linux
./run_experiment.sh quick
./run_experiment.sh all-quick
```

## How It Works
1. Config resolves from YAML + CLI.
2. Dataset generation creates pos/neg examples for training (small `n`) and testing (large `n`).
3. An RNN and a leaky SNN are built.
4. Words are fed as one-hot impulses (spikes).
5. The models are evaluated on testing word-length buckets to check generalization.

