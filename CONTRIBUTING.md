# Contributing

**English** · [中文](CONTRIBUTING_cn.md)

Bug reports and focused pull requests are welcome.

## Development

```bash
make -j
make check
```

`make check` includes a lightweight Markdown check (`scripts/check_docs.py`) for
relative links, fenced code blocks and GitHub-compatible math delimiters, plus
`scripts/check_version.py` to keep CMake, the C header and Python metadata in
sync.

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
