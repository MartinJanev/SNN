#!/bin/bash
# Convenience script for running common experiments
# Usage: ./run_experiment.sh [command]

case "$1" in
  "quick")
    echo "Running quick test..."
    python main.py --config configs/quick.yaml
    ;;
  "default")
    echo "Running default a^n b^n experiment..."
    python main.py
    ;;
  "palindrome")
    echo "Running palindrome experiment..."
    python main.py --language palindrome
    ;;
  "paren")
    echo "Running balanced parentheses experiment..."
    python main.py --language paren
    ;;
  "anbncn")
    echo "Running a^n b^n c^n experiment..."
    python main.py --language anbncn
    ;;
  "all-quick")
    echo "Running all languages with quick config..."
    for lang in anbn palindrome paren anbncn equal_ab ends_with_abb repeat_ab ww prime_a alternating; do
      echo ""
      echo "================================"
      echo "Testing: $lang"
      echo "================================"
      python main.py --language $lang --config configs/quick.yaml
    done
    ;;
  "help"|"-h"|"--help"|"")
    echo "SNN Formal Language Learning - Quick Run Script"
    echo ""
    echo "Usage: ./run_experiment.sh [command]"
    echo ""
    echo "Commands:"
    echo "  quick           - Run quick test (50 pairs, 2 epochs)"
    echo "  default         - Run default experiment (a^n b^n, 400 pairs, 5 epochs)"
    echo "  palindrome      - Test palindrome language"
    echo "  paren           - Test balanced parentheses"
    echo "  anbncn          - Test a^n b^n c^n (context-sensitive)"
    echo "  all-quick       - Run all 10 languages with quick config"
    echo "  help            - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_experiment.sh quick"
    echo "  ./run_experiment.sh palindrome"
    echo "  ./run_experiment.sh all-quick"
    echo ""
    echo "For more control, use python directly:"
    echo "  python main.py --language ww --epochs 5 --train-pairs 100"
    echo ""
    exit 0
    ;;
  *)
    echo "Unknown command: $1"
    echo "Run './run_experiment.sh help' for available commands"
    exit 1
    ;;
esac

