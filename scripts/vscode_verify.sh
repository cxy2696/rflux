#!/usr/bin/env bash
# Reproducible rflux validation used by the VS Code task
# "rflux: verify all (build + ctest + pytest + CLI)".
#
# Every step is logged separately under artifacts/validation/<timestamp>/ and a
# summary.json / summary.md is written for the run.  The script keeps going
# after a failed step so one broken area does not hide the state of the others;
# the exit status is non-zero if any required step failed.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

CLEAN=0
SANITIZE=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    --sanitize) SANITIZE=1 ;;
    -h|--help)
      echo "usage: $0 [--clean] [--sanitize]"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || true)"
fi
if [[ -z "$PYTHON" ]]; then
  echo "no Python interpreter found" >&2
  exit 2
fi

export PATH="$ROOT/.venv/bin:$PATH"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/artifacts/validation/$STAMP"
mkdir -p "$OUT"
RESULTS="$OUT/results.tsv"
: > "$RESULTS"

FAILED=0
STEP_INDEX=0

run_step() {
  local name="$1"
  shift
  STEP_INDEX=$((STEP_INDEX + 1))
  local slug
  slug="$(printf '%02d' "$STEP_INDEX")-$(echo "$name" | tr ' /()' '____' | tr -cd '[:alnum:]_-')"
  local log="$OUT/$slug.log"
  local start end seconds status
  start="$(date +%s)"
  printf '\n==> [%s] %s\n' "$STEP_INDEX" "$name"
  if "$@" >"$log" 2>&1; then
    status="passed"
  else
    status="failed"
    FAILED=1
  fi
  end="$(date +%s)"
  seconds=$((end - start))
  printf '    %s (%ss, log: %s)\n' "$status" "$seconds" "$(basename "$log")"
  if [[ "$status" == "failed" ]]; then
    tail -n 20 "$log" | sed 's/^/    | /'
  fi
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$seconds" "$slug.log" >>"$RESULTS"
}

if [[ "$CLEAN" == "1" ]]; then
  echo "removing build/dev (requested by --clean)"
  rm -rf "${ROOT:?}/build/dev"
fi

# ---------------------------------------------------------------------------
# 0. environment
# ---------------------------------------------------------------------------
run_step "native install" "$PYTHON" -m pip install -e . --no-deps
{
  echo "repository: $ROOT"
  echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname: $(uname -a)"
  echo "python: $("$PYTHON" --version 2>&1) ($PYTHON)"
  echo "cmake: $(cmake --version | head -n 1)"
  echo "compiler: $(c++ --version | head -n 1)"
  echo "ninja: $("$ROOT/.venv/bin/ninja" --version 2>/dev/null || echo 'not in venv')"
} >"$OUT/environment.txt" 2>&1
cat "$OUT/environment.txt"

run_step "python version and native info" "$PYTHON" -c '
import json, sys
import numpy as np
import rflux
info = rflux.native_info()
print("python", sys.version.split()[0])
print("rflux", rflux.__version__)
print(json.dumps(info, indent=2, default=str))
assert info.get("available"), info
assert rflux.abi_version() == 1
print("sma", rflux.qflux.sma(np.arange(6.0), 3).tolist())
'

# ---------------------------------------------------------------------------
# 1. native build and C/C++ tests
# ---------------------------------------------------------------------------
run_step "cmake configure (dev)" cmake --preset dev \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPython_EXECUTABLE="$PYTHON"
run_step "cmake build (dev)" cmake --build --preset dev
run_step "ctest (dev)" ctest --preset dev --output-on-failure

# ---------------------------------------------------------------------------
# 2. Python test suite
# ---------------------------------------------------------------------------
run_step "pytest" "$PYTHON" -m pytest tests/python -q
run_step "algorithm matrix and timings" make algo-check

# ---------------------------------------------------------------------------
# 3. command line surface
# ---------------------------------------------------------------------------
run_step "cli version" "$PYTHON" -m rflux version --json
run_step "cli doctor" "$PYTHON" -m rflux doctor
run_step "cli analyze sample" "$PYTHON" -m rflux analyze examples/data/sample_prices.csv --no-hmm --format json
run_step "cli risk sample" "$PYTHON" -m rflux risk examples/data/sample_prices.csv --weights 0.2,0.2,0.2,0.2,0.2 --format json
run_step "cli advice sample" "$PYTHON" -m rflux advice examples/data/sample_prices.csv --no-hmm --format json

# ---------------------------------------------------------------------------
# 4. examples and repository checks
# ---------------------------------------------------------------------------
run_step "examples (python + c/c++)" bash scripts/run_examples.sh
run_step "docs check" "$PYTHON" scripts/check_docs.py
run_step "version check" "$PYTHON" scripts/check_version.py

# ---------------------------------------------------------------------------
# 5. optional sanitizer build
# ---------------------------------------------------------------------------
if [[ "$SANITIZE" == "1" ]]; then
  run_step "cmake configure (sanitize)" cmake --preset sanitize
  run_step "cmake build (sanitize)" cmake --build --preset sanitize
  # macOS does not support LeakSanitizer; keep the option off there.
  if [[ "$(uname -s)" == "Darwin" ]]; then
    LEAK_OPTION="detect_leaks=0"
  else
    LEAK_OPTION="detect_leaks=1"
  fi
  run_step "ctest (sanitize)" env ASAN_OPTIONS="${LEAK_OPTION}:halt_on_error=1" \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    ctest --preset sanitize --output-on-failure
fi

# ---------------------------------------------------------------------------
# 6. summary
# ---------------------------------------------------------------------------
"$PYTHON" - "$OUT" "$RESULTS" "$FAILED" <<'PY'
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
results_path = Path(sys.argv[2])
failed = int(sys.argv[3])

rows = []
for line in results_path.read_text().splitlines():
    if not line.strip():
        continue
    name, status, seconds, log = line.split("\t")
    rows.append({"name": name, "status": status, "seconds": int(seconds), "log": log})

summary = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "passed": sum(r["status"] == "passed" for r in rows),
    "failed": sum(r["status"] == "failed" for r in rows),
    "total_seconds": sum(r["seconds"] for r in rows),
    "steps": rows,
    "ok": failed == 0,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = [
    "# rflux validation run",
    "",
    f"- Generated: {summary['generated_at']}",
    f"- Platform: {summary['platform']}",
    f"- Python: {summary['python']}",
    f"- Result: **{'PASS' if summary['ok'] else 'FAIL'}** "
    f"({summary['passed']} passed, {summary['failed']} failed, "
    f"{summary['total_seconds']}s)",
    "",
    "| # | step | status | seconds | log |",
    "| --- | --- | --- | --- | --- |",
]
for i, row in enumerate(rows, 1):
    lines.append(
        f"| {i} | {row['name']} | {row['status']} | {row['seconds']} | "
        f"[{row['log']}]({row['log']}) |"
    )
lines.append("")
(out / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))
PY

echo
echo "artifacts: $OUT"
if [[ "$FAILED" != "0" ]]; then
  echo "RESULT: FAIL"
  exit 1
fi
echo "RESULT: PASS"
