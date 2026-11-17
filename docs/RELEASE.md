# Release checklist

**English** · [中文](RELEASE_cn.md)

A release is a repository-wide contract, not just a version bump.

1. **Version sync**  
   Update `CMakeLists.txt`, `include/rflux/rflux.h`,
   `python/rflux/_version.py`, `pyproject.toml` and `docs/ABI.md`;
   `python scripts/check_version.py` must pass.
2. **C ABI**  
   Confirm `RFLUX_ABI_VERSION` only changes for a breaking layout/signature
   change.  Run the C ABI test and compare exported symbols on Linux/macOS.
3. **Native ABI policy**  
   CPython 3.12+ builds `cp312-abi3`; CPython 3.10/3.11 produce
   version-specific wheels.  Do not claim abi3 coverage below 3.12.
4. **Tests**  
   Run `make check`, `pytest tests/python`, and
   `ctest --preset dev --output-on-failure`.  Run the sanitizer preset or CI
   sanitizer job.  Provider adapter tests use mocked schemas; live provider
   smoke tests are a separate credentialed job.
5. **Data and model contracts**  
   Validate `docs/DATA_*`, `docs/NATIVE_*`, `docs/ALGORITHMS_*` and
   `docs/CORE_*`; update `CHANGELOG.md` with estimator/causality changes.
6. **Artifacts**  
   Build with the `Wheels` workflow for Linux/macOS/Windows, install each wheel
   in a clean environment and run the wheel smoke tests.  Keep the sdist and
   wheels for the tagged release.
7. **Supply chain**  
   Pin the vendored pybind11 commit and review third-party license files.
8. **Post-release**  
   Tag the exact commit and attach the sdist/wheel artifacts to the repository
   release.  This project does not publish to PyPI.
