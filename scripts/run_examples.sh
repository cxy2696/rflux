#!/usr/bin/env bash
# Run the deterministic Python and C/C++ examples that ship with rflux.
# Used by the VS Code task "rflux: examples (Python)" and by vscode_verify.sh.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

if ! "$PYTHON" -c "import rflux" >/dev/null 2>&1; then
  echo "rflux is not installed for $PYTHON; run 'make native' first" >&2
  exit 2
fi

failed=0
run() {
  echo
  echo "== $*"
  if "$@"; then
    echo "-- ok"
  else
    echo "-- FAILED"
    failed=1
  fi
}

run "$PYTHON" examples/python_algorithm_matrix.py
run "$PYTHON" examples/python_applications.py
run "$PYTHON" examples/python_factor_screen.py
run "$PYTHON" examples/python_hmm_walkforward.py
run "$PYTHON" examples/python_end_to_end.py

for exe in build/dev/example_c_basic build/dev/example_cpp_research; do
  if [[ -x "$exe" ]]; then
    run "./$exe"
  fi
done

exit "$failed"
