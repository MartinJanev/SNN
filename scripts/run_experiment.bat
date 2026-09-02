@echo off
REM Convenience script for running common experiments
REM Usage: scripts\run_experiment.bat [command]

cd /d "%~dp0\.."

if "%1"=="default" (
    echo Running default experiment (configs/config.yaml)...
    python -m formal_language_snn --config configs/config.yaml
    goto :eof
)

if "%1"=="all" (
    echo Running all languages with configs/config.yaml...
    for %%L in (anbn palindrome balanced_parens reber even_a) do (
        echo.
        echo ================================
        echo Testing: %%L
        echo ================================
        python -m formal_language_snn --config configs/config.yaml --set experiment.language=%%L
    )
    goto :eof
)

if "%1"=="help" goto :help
if "%1"=="-h" goto :help
if "%1"=="--help" goto :help
if "%1"=="" goto :help

echo Unknown command: %1
echo Run 'scripts\run_experiment.bat help' for available commands
exit /b 1

:help
echo SNN Formal Language Learning - Run Script
echo.
echo Usage: scripts\run_experiment.bat [command]
echo.
echo Commands:
echo   default        - Run experiment from configs/config.yaml
echo   all            - Run all languages using configs/config.yaml
echo   help           - Show this help message
echo.
echo Examples:
echo   scripts\run_experiment.bat default
echo   scripts\run_experiment.bat all
echo.
echo For more control, use python directly:
echo   python -m formal_language_snn --config configs/config.yaml --set experiment.language=palindrome --set training.epochs=10
exit /b 0
