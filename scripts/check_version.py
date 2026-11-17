#!/usr/bin/env python3
"""Check that every duplicated version declaration stays in sync."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _match(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise ValueError(f"could not parse {label}")
    return match.group(1)


def main() -> int:
    try:
        cmake = _match(
            r"project\(\s*rflux\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)",
            _read("CMakeLists.txt"),
            "CMakeLists.txt",
        )
        header = _read("include/rflux/rflux.h")
        major = _match(r"#define\s+RFLUX_VERSION_MAJOR\s+(\d+)", header, "rflux.h")
        minor = _match(r"#define\s+RFLUX_VERSION_MINOR\s+(\d+)", header, "rflux.h")
        patch = _match(r"#define\s+RFLUX_VERSION_PATCH\s+(\d+)", header, "rflux.h")
        python_version = _match(
            r'^__version__\s*=\s*"([^"]+)"',
            _read("python/rflux/_version.py"),
            "python/rflux/_version.py",
        )
        abi_doc = _read("docs/ABI.md")
        doc_major = _match(
            r"#define\s+RFLUX_VERSION_MAJOR\s+(\d+)", abi_doc, "docs/ABI.md"
        )
        doc_minor = _match(
            r"#define\s+RFLUX_VERSION_MINOR\s+(\d+)", abi_doc, "docs/ABI.md"
        )
        doc_patch = _match(
            r"#define\s+RFLUX_VERSION_PATCH\s+(\d+)", abi_doc, "docs/ABI.md"
        )
        pyproject = _match(
            r'^version\s*=\s*"([^"]+)"',
            _read("pyproject.toml"),
            "pyproject.toml",
        )
    except (OSError, ValueError) as exc:
        print(f"version-check failed: {exc}", file=sys.stderr)
        return 1

    versions = {
        "CMakeLists.txt": cmake,
        "include/rflux/rflux.h": f"{major}.{minor}.{patch}",
        "python/rflux/_version.py": python_version,
        "pyproject.toml": pyproject,
        "docs/ABI.md": f"{doc_major}.{doc_minor}.{doc_patch}",
    }
    unique = set(versions.values())
    if len(unique) != 1:
        print("version-check failed: declarations disagree", file=sys.stderr)
        for name, value in versions.items():
            print(f"  {name}: {value}", file=sys.stderr)
        return 1

    print(f"version-check: {unique.pop()} ({len(versions)} declarations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
