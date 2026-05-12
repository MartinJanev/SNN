@echo off
REM Convenience script for running common experiments
REM Usage: run_experiment.bat [command]

if "%1%"=="" goto help
if "%1%"=="help" goto help
if "%1%"=="-h" goto help
if "%1%"=="--help" goto help

if "%1%"=="quick" (
    echo Running quick test...
    python main.py --config configs/quick.yaml
    goto end
)

if "%1%"=="default" (
    echo Running default a^n b^n experiment...
    python main.py
    goto end
)

if "%1%"=="palindrome" (
    echo Running palindrome experiment...
    python main.py --language palindrome
    goto end
)

if "%1%"=="paren" (
    echo Running balanced parentheses experiment...
    python main.py --language paren
    goto end
)

if "%1%"=="anbncn" (
    echo Running a^n b^n c^n experiment...
    python main.py --language anbncn
    goto end
)

if "%1%"=="all-quick" (
    echo Running all languages with quick config...
    for %%L in (anbn palindrome paren anbncn equal_ab ends_with_abb repeat_ab ww prime_a alternating) do (
        echo.
        echo ================================
        echo Testing: %%L
        echo ================================
        python main.py --language %%L --config configs/quick.yaml
    )
    goto end
)

echo Unknown command: %1%
echo Run 'run_experiment.bat help' for available commands
goto end

:help
echo SNN Formal Language Learning - Quick Run Script
echo.
echo Usage: run_experiment.bat [command]
echo.
echo Commands:
echo   quick           - Run quick test (50 pairs, 2 epochs)
echo   default         - Run default experiment (a^n b^n, 400 pairs, 5 epochs)
echo   palindrome      - Test palindrome language
echo   paren           - Test balanced parentheses
echo   anbncn          - Test a^n b^n c^n (context-sensitive)
echo   all-quick       - Run all 10 languages with quick config
echo   help            - Show this help message
echo.
echo Examples:
echo   run_experiment.bat quick
echo   run_experiment.bat palindrome
echo   run_experiment.bat all-quick
echo.
echo For more control, use python directly:
echo   python main.py --language ww --epochs 5 --train-pairs 100
echo.

:end

