#!/bin/bash
# Convenience script for running common experiments
# Usage: ./scripts/run_experiment.sh [command]

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

case "$1" in
  "default")
    echo "Running default experiment (configs/config.yaml)..."
    python -m formal_language_snn --config configs/config.yaml
    ;;
  "all")
    echo "Running all languages with configs/config.yaml..."
    for lang in anbn palindrome balanced_parens reber even_a; do
      echo ""
      echo "================================"
      echo "Testing: $lang"
      echo "================================"
      python -m formal_language_snn --config configs/config.yaml --set "experiment.language=$lang"
    done
    ;;
  "help"|"-h"|"--help"|"")
    echo "SNN Formal Language Learning - Run Script"
    echo ""
    echo "Usage: ./scripts/run_experiment.sh [command]"
    echo ""
    echo "Commands:"
    echo "  default        - Run experiment from configs/config.yaml"
    echo "  all            - Run all languages using configs/config.yaml"
    echo "  help           - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_experiment.sh default"
    echo "  ./scripts/run_experiment.sh all"
    echo ""
    echo "For more control, use python directly:"
    echo "  python -m formal_language_snn --config configs/config.yaml --set experiment.language=palindrome --set training.epochs=10"
    echo ""
    exit 0
    ;;
  *)
    echo "Unknown command: $1"
    echo "Run './scripts/run_experiment.sh help' for available commands"
    exit 1
    ;;
esac
