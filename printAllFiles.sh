#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_FILE="${1:-$SCRIPT_DIR/all_contents.txt}"
ROOT_DIR="${2:-$SCRIPT_DIR}"
MAX_BYTES="${MAX_BYTES:-1048576}"

ROOT_DIR="$(cd "$ROOT_DIR" && pwd)"
if [[ "$OUT_FILE" != /* ]]; then
  OUT_FILE="$PWD/$OUT_FILE"
fi
mkdir -p "$(dirname "$OUT_FILE")"

should_skip_dir() {
  case "$1" in
    */.git|*/.git/*|*/outputs/*|*/.venv|*/.venv/*|*/__pycache__|*/__pycache__/*|*/.idea|*/.idea/*|*/.mypy_cache|*/.mypy_cache/*|*/.pytest_cache|*/.pytest_cache/*|*/.cache|*/.cache/*|*/cache|*/cache/*|*/node_modules|*/node_modules/*|*/output|*/output/*|*/output_cpu|*/output_cpu/*|*/dist|*/dist/*|*/build|*/build/*|*output_so_relevant_data|*output_so_relevant_data/*)
      return 0
      ;;
  esac
  return 1
}

should_skip_file() {
  case "$1" in
    "$OUT_FILE"|*.pyc|*.pyo|*.parquet|*.csv|*.csv.gz|*.zip|*.gz|*.png|*.jpg|*.jpeg|*.gif|*.pdf|*.so|*.dylib|*.o|*.obj|*.class)
      return 0
      ;;
  esac
  return 1
}

: >"$OUT_FILE"
{
  echo "# printallfiles output"
  echo "# root: $ROOT_DIR"
  echo "# generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "# max_bytes: $MAX_BYTES"
  echo
} >>"$OUT_FILE"

while IFS= read -r -d '' file; do
  [[ -f "$file" ]] || continue
  rel_path="${file#"$ROOT_DIR"/}"

  if should_skip_dir "$file" || should_skip_file "$file"; then
    continue
  fi

  if [[ ! -s "$file" ]]; then
    {
      echo "===== FILE: $rel_path ====="
      echo ""
    } >>"$OUT_FILE"
    continue
  fi

  file_size=$(wc -c <"$file" | tr -d '[:space:]')
  if [[ "$file_size" -gt "$MAX_BYTES" ]]; then
    continue
  fi

  {
    echo "===== FILE: $rel_path ====="
    cat "$file"
    echo
  } >>"$OUT_FILE"
done < <(
  find "$ROOT_DIR" \
    \( -type d \( \
      -name .git -o -name .venv -o -name __pycache__ -o -name .idea -o -name .mypy_cache -o -name .pytest_cache -o -name .cache -o -name cache -o -name node_modules -o -name output -o -name output_cpu -o -name dist -o -name build \
    \) -prune \) -o \
    -type f -print0
)

echo "Wrote $OUT_FILE"
