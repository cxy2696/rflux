# Native Python bridge, zero-copy and abi3

**English** · [中文](NATIVE_cn.md)

rflux keeps three independent layers:

```
C ABI (rflux.h)        other languages / long-term binary contract
C++17 core             pure math, no file I/O or Python types
pybind11 extension     NumPy buffers, GIL release, stable ABI
Python research layer  validation, data adapters, CLI, configuration
```

The extension is optional at the source level. When it is present the public
`qflux` functions and `regimelab.GaussianHMM` use it automatically; when
it is absent the package falls back to the stable C ABI through `ctypes`.
`RFLUX_DISABLE_NATIVE=1` forces the fallback for tests and diagnostics.

```python
import rflux
print(rflux.native_info())
# {'available': True, 'version': '0.7.0', 'stable_abi': True,
#  'gil_released': True, 'module': '.../_native.abi3.so'}
```

## Zero-copy buffers

The native functions accept C-contiguous `float64` NumPy arrays through
pybind11's buffer-aware array caster. A contiguous array is passed to C++
without a data copy; a non-contiguous or non-`float64` input is cast first by
NumPy. Outputs are new arrays, except for the explicit `out=`-style entry point:

```python
import numpy as np
from rflux import native, qflux

values = np.arange(20.0)
out = np.empty_like(values)
result = native.series_into("rflux_sma", values, out, 3)
assert result is out
np.testing.assert_allclose(out, qflux.sma(values, 3), equal_nan=True)
```

`series_into` rejects a non-contiguous, wrong-dtype or read-only output buffer,
so a copy can never silently hide the zero-copy contract. The HMM `fit` path
passes the observation panel to C++ in the same way; the C++ side validates
shape, finiteness and feature count before entering EM.

The same mechanism backs the 0.7 algorithm modules: factor panels, HRP weights,
Almgren-Chriss schedules and Huber regression are exposed through the native
module and release the GIL while the C++ kernel runs.

For inter-process transfer use `rflux.SharedArray`:

```python
from rflux.shared import SharedArray
import numpy as np

with SharedArray.create((1_000_000,), unlink_on_exit=True) as panel:
    panel.array[:] = np.random.default_rng(0).normal(size=panel.array.size)
    # Pass panel.array to the native bridge, or attach to the same segment
    # from another process with SharedArray.attach(panel.name, (1_000_000,)).
    print(panel.name, panel.array.nbytes)
```

Only the small descriptor is pickled; the payload stays in the OS shared-memory
segment. The creator should call `unlink()` after all consumers detach. A
segment is page-rounded by the operating system, so `attach` accepts a shape
whose payload fits the allocated segment.

## GIL release

All expensive native entry points release the GIL around the C++ section:
HMM fitting, scoring, filtering, smoothing, Viterbi, and the qflux
kernels. Validation and array allocation stay on the Python side. This
allows an ingestion or networking thread to keep running while a large
Baum-Welch fit is in progress.

## Stable ABI (abi3)

CPython 3.12+ wheels are built with pybind11's `STABLE_ABI` mode. One
`_native.abi3.so` therefore serves CPython 3.12, 3.13 and later releases. The
wheel tag is `cp312-abi3-...`; it does not contain a per-interpreter binary for
each minor version.

CPython 3.10 and 3.11 do not have the stable ABI subset pybind11 needs, so those
wheels are version-specific (`cp310`, `cp311`). The Python package still
supports all four versions; only the binary tag differs.

The stable ABI support is currently vendored from pybind11 commit
`33e4318d64e795cb32a13836da8830e9cd42ab5f` (`3.2.0a0`) in
`third_party/pybind11`, because the feature is not in a PyPI release yet. The
commit is pinned and the vendored tree is self-contained, so offline wheel
builds do not depend on a network fetch. Replace it wholesale when pybind11 3.2
is released.

## Configuration-driven models

`rflux.config` loads TOML, JSON or YAML at the composition edge and instantiates
the pure C++ model through its parameter interface. The core never parses files
or depends on a configuration library.

```toml
[model]
type = "hmm"
n_states = 2
n_features = 1

[model.initial]
start_prob = [0.6, 0.4]
transition = [[0.9, 0.1], [0.2, 0.8]]
means = [[-1.0], [1.0]]
variances = [[1.0], [1.0]]

[model.fit]
max_iter = 200
n_init = 3
seed = 42
```

```python
from rflux import config

model_config = config.load("model.toml").model
model = model_config.build()          # parameters loaded, not fitted
fitted = model_config.fit_model(returns)
```

`n_states` and `n_features` may be omitted when `model.initial` exists; they are
inferred and checked against the tables. Unknown keys, non-finite parameters,
negative variances and shape mismatches fail with `rflux.ConfigError` before a
native handle is created. `rflux config model.toml --json --build` provides the
same validation from the CLI.

## Build and verification

The native path requires CMake 3.26+ and Ninja. A system GoogleTest is used when
present; pass `-DRFLUX_FETCH_GTEST=ON` to download it instead.

```bash
python -m pip install ".[test]"
python -m pytest tests/python -q

# Optional CMake/Ninja developer loop
cmake --preset dev
cmake --build --preset dev
ctest --preset dev --output-on-failure

# Build a local wheel
python -m pip wheel . --no-deps -w dist
```

`python -m rflux doctor --json` reports both the C ABI shared library and the
pybind11 extension, including whether the extension was built with the stable
ABI and whether GIL release is enabled.

## Design notes

- The extension links the same `rflux_core` static library as the C ABI; no
  numerical logic is duplicated.
- The C ABI shared library is also shipped inside the wheel at
  `rflux/_lib/`, so the `ctypes` fallback and the CLI keep working in a clean
  installation.
- Native exceptions carry the same status codes as the C ABI
  (`1` invalid argument, `3` non-finite, `4` model not ready, `5` numerical).
  The Python layer translates them back to `rflux.RFluxError`.
- All outputs are checked for shape before the C++ call; no Python API is used
  while the GIL is released.
