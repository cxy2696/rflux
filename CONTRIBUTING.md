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
sync, and `make algo-check`, which runs `scripts/verify_algorithms.py` once
through the native extension and once through the ctypes C ABI.  The algorithm
traceability matrix lives in `docs/ALGORITHMS_en.md`.

The shipping code and tests are kept at zero exact clones. Run `make dup-check`
before a large refactor (it needs Node/`npx`, so CI does not run it). The
command checks `src/`, `include/`, `python/rflux/` and `tests/` at the strict
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

# ASan + UBSan (Linux CI runs leak detection; macOS cannot)
cmake --preset sanitize
cmake --build --preset sanitize
ctest --preset sanitize --output-on-failure
```

The CMake build uses a system GoogleTest when present. If it is not installed,
pass `-DRFLUX_FETCH_GTEST=ON` to download and build it, or leave
`-DRFLUX_BUILD_GTEST=OFF` to skip only that suite.

`RFLUX_DISABLE_NATIVE=1` forces the ctypes fallback. When adding a native
binding, release the GIL around expensive C++ work and keep all Python API calls
outside the released region. New public behaviour needs a pytest under
`tests/python/`; new C++ kernels should also get a GoogleTest case.

## Adding a kernel

1. Declare it in the relevant `include/rflux/**/*.hpp` header.
2. Implement it under `src/`.
3. Add a numeric test with known inputs, including NaN/zero-variance/edge cases where relevant.
4. If it needs to be used from other languages, add a C function in `include/rflux/rflux.h` and `src/rflux_capi.cpp`, then a ctypes binding and a C ABI test.
5. For deterministic graph kernels, test known graphs and compare against a brute-force implementation on a small random graph. For Monte Carlo kernels, verify the mean and population standard deviation against independent single-replicate calls.

## Adding a C ABI function

- Return `rflux_status` and use `api_guard` in `src/rflux_capi.cpp`.
- Document the output length and missing-value behavior above the declaration.
- Do not expose C++ types or let exceptions cross the boundary.
- Existing ABI-v1 function signatures and option layouts must stay unchanged. A breaking change requires a new ABI version.

## Python changes

Keep optional data dependencies lazy. `import rflux` must not access the network or require pandas. New public functions need a test under `tests/python/`.

## Style

Match the surrounding code. C++ code is C++17, Python targets 3.10+. Prefer clear names and explicit edge-case handling over abstractions that hide numerical behavior.
