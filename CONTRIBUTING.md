# Contributing

**English** · [中文](CONTRIBUTING_cn.md)

Bug reports and focused pull requests are welcome.

## Development

```bash
make -j
make check
```

For an end-to-end local verification (build, CTest, pytest, CLI, examples and
docs), run `bash scripts/vscode_verify.sh`. It writes logs under
`artifacts/validation/` and exits non-zero on the first failed area.

`make check` includes a lightweight Markdown check (`scripts/check_docs.py`) for
relative links, fenced code blocks and GitHub-compatible math delimiters,
`scripts/check_version.py` to keep CMake, the C header and Python metadata in
sync, and `make algo-check`, which runs `scripts/verify_algorithms.py` through
the native extension.  The algorithm traceability matrix lives in
`docs/ALGORITHMS_en.md`.

The shipping code and tests are kept at zero exact clones. Run `make dup-check`
before a large refactor (it needs Node/`npx`, so CI does not run it). GitHub
Actions runs on `ubuntu-latest` only; build and test locally on other platforms
when changing the C ABI or bindings. The command checks `src/`, `include/`, `python/rflux/` and `tests/` at the strict
5-line/40-token threshold; the bilingual documentation mirrors are the only
expected repeated blocks.

## Documentation style

Write documentation as engineering notes for someone who has to reproduce the
result. State the input, output, estimator and test in direct sentences. Remove
marketing adjectives, repeated disclaimers and generated-sounding transitions;
state a limitation once, where it affects use. Keep the English and Chinese
pages in sync. `scripts/check_docs.py` rejects a short list of generated-sounding
markers such as `not just`, `deliberately` and `Next layer`.

The Make build is the quickest loop for the C/C++ surface. For the native
Python extension and the reproducible CMake/Ninja loop:

```bash
python -m pip install -e ".[test]"
python -m pytest tests/python -q

cmake --preset dev
cmake --build --preset dev
ctest --preset dev --output-on-failure

# ASan + UBSan (CI enables leak detection on ubuntu-latest)
cmake --preset sanitize
cmake --build --preset sanitize
ctest --preset sanitize --output-on-failure
```

The CMake build uses a system GoogleTest when present. If it is not installed,
pass `-DRFLUX_FETCH_GTEST=ON` to download and build it, or leave
`-DRFLUX_BUILD_GTEST=OFF` to skip only that suite.

New public behaviour needs a pytest under `tests/python/`; new C++ kernels
should also get a GoogleTest case. Release the GIL around expensive C++ work
and keep all Python API calls outside the released region. Algorithmic work and
computation belong in C++; Python should only validate inputs and assemble
reports.

## Adding a kernel

1. Declare it in the relevant `include/rflux/**/*.hpp` header.
2. Implement it under `src/`.
3. Add a numeric test with known inputs, including NaN/zero-variance/edge cases where relevant.
4. Bind it through pybind11 in `src/python/bindings.cpp`. If C/C++ callers need the same kernel, also add a C function in `include/rflux/rflux.h` and `src/rflux_capi.cpp` with a C ABI test.
5. For deterministic graph kernels, test known graphs and compare against a brute-force implementation on a small random graph. For Monte Carlo kernels, verify the mean and population standard deviation against independent single-replicate calls.

## Adding a C ABI function

- Return `rflux_status` and use `api_guard` in `src/rflux_capi.cpp`.
- Document the output length and missing-value behavior above the declaration.
- Do not expose C++ types or let exceptions cross the boundary.
- Existing ABI-v1 function signatures and option layouts must stay unchanged. A breaking change requires a new ABI version.

## Python changes

Keep optional data dependencies lazy. `import rflux` must not access the network or require pandas. New public functions need a test under `tests/python/`.

## Native extension

The pybind11 module is required: the package has no pure-Python execution path. Build it with `python -m pip install -e .` or `make native`; `cmake --preset dev` produces a developer extension in `build/dev`. Release the GIL around expensive C++ work, keep all Python API calls outside the released region, and let `*_into` kernels fill NumPy output buffers directly. The wheel uses abi3 on CPython 3.12+ and version-specific wheels on 3.10/3.11; see `docs/ABI.md` for the binary contract.

## More extension points

- Data source: implement the loader contract in `python/rflux/data.py` and add a mocked-schema test; keep provider SDK imports lazy.
- HMM variant: keep filtering causal, validate parameter tables, and add a known-state likelihood/forecast test under `tests/python/test_hmm_regime.py`.
- Graph or percolation kernel: validate symmetry and weights in C++, then compare small random graphs with a brute-force reference.

## Release

1. Update the version in `CMakeLists.txt`, `include/rflux/rflux.h`, `python/rflux/_version.py`, `pyproject.toml` and the ABI docs; run `python scripts/check_version.py`.
2. Run `make check`, `pytest tests/python`, and `ctest --preset dev --output-on-failure`; run the sanitizer preset when C++ code changed.
3. Build the sdist and wheel, install the wheel in a clean environment, and run the import/CLI smoke tests.
4. Tag the exact commit, attach the current sdist and wheels, and do not publish to PyPI.

## Style

Match the surrounding code. C++ code is C++17, Python targets 3.10+. Prefer clear names and explicit edge-case handling over abstractions that hide numerical behavior.
