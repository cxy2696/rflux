# Contributing

Bug reports and focused pull requests are welcome.

## Development

```bash
make -j
make check
```

`make check` includes a lightweight Markdown check (`scripts/check_docs.py`) for
relative links, fenced code blocks and GitHub-compatible math delimiters.

The Make build is the quickest loop. If CMake is available, also run:

```bash
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake -j
ctest --test-dir build/cmake --output-on-failure
```

## Adding a kernel

1. Declare it in the relevant `include/rflux/**/*.hpp` header.
2. Implement it under `src/`.
3. Add a numeric test with known inputs, including NaN/zero-variance/edge cases where relevant.
4. If it needs to be used from other languages, add a C function in `include/rflux/rflux.h` and `src/rflux_capi.cpp`, then a ctypes binding and a C ABI test.

## Adding a C ABI function

- Return `rflux_status` and use `api_guard` in `src/rflux_capi.cpp`.
- Document the output length and missing-value behavior above the declaration.
- Do not expose C++ types or let exceptions cross the boundary.
- Existing ABI-v1 function signatures and option layouts must stay unchanged. A breaking change requires a new ABI version.

## Python changes

Keep optional data dependencies lazy. `import rflux` must not access the network or require pandas. New public functions need a test under `tests/python/`.

## Style

Match the surrounding code. C++ code is C++17, Python targets 3.10+. Prefer clear names and explicit edge-case handling over abstractions that hide numerical behavior.
