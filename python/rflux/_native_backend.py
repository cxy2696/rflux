"""Access to the compiled rflux Python extension.

The package requires the pybind11 extension: every algorithm and computation
runs in the C++ core.  Build it with ``python -m pip install -e .`` (or
``cmake --preset dev`` for a local developer build).
"""

from __future__ import annotations

from typing import Any, Callable, NoReturn, TypeVar

from ._version import __version__

STATUS_NAMES = {
    0: "RFLUX_OK",
    1: "RFLUX_ERR_INVALID_ARGUMENT",
    2: "RFLUX_ERR_OUT_OF_MEMORY",
    3: "RFLUX_ERR_NONFINITE",
    4: "RFLUX_ERR_MODEL_NOT_READY",
    5: "RFLUX_ERR_NUMERICAL",
    6: "RFLUX_ERR_INTERNAL",
}


class RFluxError(RuntimeError):
    """A native call returned a non-zero rflux_status."""

    def __init__(self, function: str, status: int, message: str):
        self.function = function
        self.status = int(status)
        prefix = f"{function}: "
        if message.startswith(prefix):
            message = message[len(prefix):]
        self.message = message
        name = STATUS_NAMES.get(self.status, f"RFLUX_STATUS_{self.status}")
        detail = f": {message}" if message else ""
        super().__init__(f"{function} failed with {name}{detail}")


class ABIError(ImportError):
    """The native extension and the Python package disagree on the version."""


try:
    from . import _native as native_module
except ImportError as exc:  # pragma: no cover - depends on build state
    raise ImportError(
        "the rflux native extension is required but was not built; run "
        "`python -m pip install -e .` or `cmake --preset dev`"
    ) from exc

_extension_version = getattr(native_module, "__version__", None)
if _extension_version != __version__:
    raise ABIError(
        f"native extension version {_extension_version!r} does not match "
        f"Python package {__version__}"
    )

NATIVE_AVAILABLE = True
ABI_VERSION = int(native_module.ABI_VERSION)
T = TypeVar("T")


class _NativeApi:
    """Translate ``NativeError`` to ``RFluxError`` for Python wrappers.

    ``native_module`` stays the raw extension object for tests and the
    low-level ``rflux.native`` facade; high-level modules call this proxy so
    invalid arguments report the public exception type.
    """

    def __init__(self, module: Any):
        self._module = module

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._module, name)
        if not callable(value) or isinstance(value, type):
            return value

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return value(*args, **kwargs)
            except self._module.NativeError as exc:
                raise_as_rflux(name, exc)

        wrapped.__name__ = name
        return wrapped


def raise_as_rflux(function: str, exc: BaseException) -> NoReturn:
    """Translate a native bridge error into the public exception type."""
    status = int(getattr(exc, "status", 6))
    raise RFluxError(function, status, str(exc)) from None


def native_call(
    module: Any, function: str, call: Callable[[], T]
) -> T:
    """Run one native call and translate ``NativeError`` to ``RFluxError``."""
    try:
        return call()
    except module.NativeError as exc:
        raise_as_rflux(function, exc)


native_api = _NativeApi(native_module)


def native_info() -> dict[str, object]:
    return {
        "available": True,
        "version": native_module.__version__,
        "stable_abi": bool(native_module.STABLE_ABI),
        "gil_released": bool(native_module.GIL_RELEASED),
        "module": str(native_module.__file__),
    }


def abi_version() -> int:
    return int(native_module.ABI_VERSION)


def native_version() -> str:
    return str(native_module.__version__)


__all__ = [
    "ABIError",
    "ABI_VERSION",
    "NATIVE_AVAILABLE",
    "RFluxError",
    "abi_version",
    "native_api",
    "native_call",
    "native_info",
    "native_module",
    "native_version",
    "raise_as_rflux",
]
