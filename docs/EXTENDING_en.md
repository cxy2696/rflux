# Extension guide

[中文](EXTENDING_cn.md) · [Back to English guide](../README_en.md)

rflux has three small boundaries: the C++ core does the computation, the C ABI carries it across languages, and Python handles argument conversion and composition. Decide which layer a new feature belongs to before writing code.

## 1. Add a C++ kernel

Using `qflux` as an example:

1. Declare the function in `include/rflux/qflux/<module>.hpp`.
2. Implement it in `src/qflux/<module>.cpp`. Use `require`, `require_pointer` and `require_finite` from `src/internal/common.hpp`, and throw `rflux::InvalidArgument`, `rflux::NonFinite` or `rflux::NumericalError`.
3. Document the details that users actually need:
   - does the input allow NaN;
   - output length and layout;
   - warm-up period;
   - behaviour for zero variance, short samples and all-missing input;
   - random seeds and determinism.
4. Add a numeric test to `tests/cpp/test_cpp.cpp`. Cover a normal case, an edge case and a degenerate case.
5. If the kernel is needed from other languages, expose it through the C ABI in the next section. Do not put `std::vector`, exceptions or internal pointers in the C boundary.

The Makefile collects `.cpp` files from the three module directories with wildcards. If you add a subdirectory, update both `Makefile` and `CMakeLists.txt`.

## 2. Add a C ABI function

### Header

Add to `include/rflux/rflux.h`:

```c
RFLUX_API rflux_status rflux_my_function(
    const double* input, size_t n,
    double* output);
```

Rules:

- Return `rflux_status` and mark the function `RFLUX_API`.
- No C++ types and no exceptions.
- State the length of every output. Reject `NULL` unless the contract explicitly allows it.
- Do not modify an existing struct layout. For new options, create a versioned options type with `struct_size` as its first field.
- Append new enum values at the end and document how older clients should handle them.

### Implementation

Wrap the kernel in `src/rflux_capi.cpp`:

```cpp
rflux_status rflux_my_function(const double* input, size_t n, double* output) {
    return api_guard("rflux_my_function", [&] {
        require_output(output, "output");
        copy_out(rflux::qflux::my_function(input, n), output);
    });
}
```

- Every operation goes inside `api_guard`, so exceptions become status codes.
- Validate output pointers before doing work; do not return partial results on failure.
- Use `checked_mul` for `size_t` products.
- Do not return internal static containers or C++ objects.
- Independent pure functions can run concurrently. Do not fit and infer on the same HMM handle at the same time.

### Tests

Add cases to `tests/c/test_capi.c`:

- normal input;
- null pointers and zero length;
- non-finite input;
- window boundaries and warm-up;
- after a failure, does a successful call clear `rflux_last_error()`;
- sentinel values past the output array to catch overruns.

Add the matching test under `tests/python/`. Every new prototype in `_ffi.py` needs complete `argtypes` and `restype`; otherwise pointers can be truncated on 64-bit or Windows.

## 3. Add a Python binding

`_configure()` in `python/rflux/_ffi.py` is the one place for ctypes signatures:

```python
bind("rflux_my_function", ctypes.c_int,
     [c_double_p, size_t, c_double_p])
```

Then convert inputs, allocate outputs and call `check(...)` in the relevant module:

- pass contiguous arrays;
- validate shapes and obvious range errors in Python, but keep validation in C as well;
- never return partially filled arrays after a failure;
- return structured results as frozen dataclasses;
- do not expose ctypes pointers.

The Python layer should own interface and composition, not duplicate numerical work.

## 4. Add a data source

Add a loader in `python/rflux/data.py`:

1. Import the provider lazily so `import rflux` still works without akshare or yfinance.
2. Raise `MissingDependencyError` with an install command when the provider is absent.
3. Normalise output to a wide `DataFrame` with a unique, ascending date index and one price column per instrument.
4. Wrap network, schema and empty-data failures in `DataSourceError`, preserving the original cause.
5. Never write API keys, cookies or tokens to logs, exceptions or the repository.
6. Keep the default test suite offline. Live-provider tests must be opt-in and skippable.

The existing CSV, akshare and yfinance loaders are the templates.

## 5. Add an HMM variant

The current `GaussianHMM` uses diagonal Gaussian emissions. For full covariance, Student-t emissions or another model:

- keep the existing `GaussianHMM` parameter layout unchanged;
- add a new type, or extend a versioned options object and the C ABI together;
- preserve the semantics: `filtered()` is causal, while `smoothed()` and `viterbi()` use the full sample;
- add numerical tests with long series, extreme values and zero-variance features;
- revisit state ordering, parameter validation and the alignment cost;
- do not silently change what Python's `GaussianHMM` means. Add a new class or an explicit model-type field.

## 6. Add a graph or percolation algorithm

Graph functions share these input rules:

- row-major n×n matrices, finite and symmetric;
- the diagonal is ignored;
- any non-negativity requirement must match across header, implementation and docs;
- random procedures take an explicit seed and are reproducible on the same platform and version;
- when labels are `int32_t`, check that n fits in `INT32_MAX`;
- Monte Carlo results explain the sample count, standard deviation, and whether the largest cluster is excluded.

If a metric needs directed edges, add a separate type or parameter. Do not quietly treat a symmetric matrix as directed.

## 7. Before committing

```bash
make clean
make -j
make check
```

`make check` covers C ABI, C++, Python and Markdown checks. You can run the pieces separately:

```bash
make docs-check
cc -std=c11 -fsyntax-only -Iinclude -x c include/rflux/rflux.h
```

Inspect exported symbols:

```bash
nm -gU build/librflux.dylib   # macOS
nm -D build/librflux.so       # Linux
```

Before a release, confirm:

1. version macros in `include/rflux/rflux.h`, the READMEs and changelog agree;
2. no ABI-breaking change shipped under the same ABI version;
3. an older C client still links against the new library;
4. new random procedures have fixed-seed tests;
5. `import rflux` does not access the network;
6. new formulas use GitHub `$...$` / `$$...$$` and `make docs-check` passes.

CI runs Make, CMake, Python tests and the examples on Ubuntu and macOS. For documentation changes, also preview the rendered page or render it through the GitHub Markdown API.
