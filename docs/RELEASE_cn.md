# 发布清单

[English](RELEASE.md) · **中文**

发布是仓库级契约，不只是改版本号。

1. **版本同步**  
   更新 `CMakeLists.txt`、`include/rflux/rflux.h`、
   `python/rflux/_version.py`、`pyproject.toml` 和 `docs/ABI_cn.md`；
   `python scripts/check_version.py` 必须通过。
2. **C ABI**  
   确认 `RFLUX_ABI_VERSION` 只在布局/签名破坏性变化时提升。运行 C ABI 测试，
   在 Linux/macOS 比较导出符号。
3. **原生 ABI 策略**  
   CPython 3.12+ 构建 `cp312-abi3`；3.10/3.11 构建版本相关 wheel。不要声称
   3.12 以下支持 abi3。
4. **测试**  
   运行 `make check`、`pytest tests/python` 和
   `ctest --preset dev --output-on-failure`；运行 sanitizer preset 或 CI
   sanitizer job。数据源适配器测试使用 mock schema；真实数据 smoke test 是
   独立的带凭据任务。
5. **数据与模型契约**  
   校验 `docs/DATA_*`、`docs/NATIVE_*`、`docs/ALGORITHMS_*` 和
   `docs/CORE_*`；把估计量/因果性变化写入 `CHANGELOG_cn.md`。
6. **产物**  
   用 `Wheels` workflow 构建 Linux/macOS/Windows wheel，在干净环境安装并运行
   smoke test。保留 tag 对应的 sdist 和 wheel。
7. **供应链**  
   固定 vendored pybind11 commit，检查第三方许可证。
8. **发布后**  
   给准确 commit 打 tag，把 sdist/wheel 附到仓库 release。本项目不发布到
   PyPI。
