#!/usr/bin/env python3
"""Cheap repository checks that catch common GitHub Markdown breakage."""

from __future__ import annotations

import re
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", "build", "__pycache__", ".venv", "venv", "dist", ".pytest_cache", "artifacts"}
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)")
STYLE_MARKERS = {
    "deliberately": "state the reason directly",
    "intentionally": "state the reason directly",
    "genuinely": "remove the intensifier",
    "transparent": "describe the mechanism instead",
    "auditable": "use documented/reproducible instead",
    "not just": "state the scope directly",
    "Next layer": "use Future work",
    "后续层": "改用“后续工作”",
    "真正的": "删除强调词",
    "可审计": "改为“文档化”或“可复现”",
    "透明": "直接描述机制",
    "目标不是继续": "直接说明该层提供什么",
}


def markdown_files() -> list[Path]:
    result = []
    for path in ROOT.rglob("*.md"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        result.append(path)
    return sorted(result)


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    fence = ""
    fence_start = 0
    relative_links = 0

    for number, line in enumerate(lines, start=1):
        match = FENCE_RE.match(line)
        if match:
            marker = match.group(2)
            if not fence:
                fence = marker[0]
                fence_start = number
            elif marker[0] == fence:
                fence = ""
                fence_start = 0
            continue
        if fence:
            continue

        plain = INLINE_CODE_RE.sub("", line)
        for marker, hint in STYLE_MARKERS.items():
            if marker in plain:
                errors.append(
                    f"{path.relative_to(ROOT)}:{number}: avoid {marker!r}; {hint}"
                )
        if re.search(r"\\[\(\[]", plain):
            errors.append(
                f"{path.relative_to(ROOT)}:{number}: use $...$ / $$...$$, "
                "not \\(...\\) / \\[...\\], for GitHub math"
            )
        for math_match in INLINE_MATH_RE.finditer(plain):
            before = plain[math_match.start() - 1] if math_match.start() else " "
            after_index = math_match.end()
            after = plain[after_index] if after_index < len(plain) else " "
            before_ok = before.isspace() or (
                ord(before) < 128 and not before.isalnum() and before != "_"
            )
            after_ok = after.isspace() or not after.isalnum()
            if not before_ok:
                errors.append(
                    f"{path.relative_to(ROOT)}:{number}: inline math needs "
                    "an ASCII space or ASCII punctuation before the opening $"
                )
            if not after_ok:
                errors.append(
                    f"{path.relative_to(ROOT)}:{number}: inline math needs "
                    "a space or punctuation after the closing $"
                )

        for target in LINK_RE.findall(plain):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative_links += 1
            clean = urllib.parse.unquote(target.split("#", 1)[0])
            if not clean:
                continue
            if not (path.parent / clean).exists():
                errors.append(
                    f"{path.relative_to(ROOT)}:{number}: missing link target "
                    f"{target!r}"
                )

    if fence:
        errors.append(
            f"{path.relative_to(ROOT)}:{fence_start}: unclosed fenced code block"
        )
    return errors


def main() -> int:
    files = markdown_files()
    errors: list[str] = []
    for path in files:
        errors.extend(check_file(path))
    if errors:
        print("documentation check failed:", file=sys.stderr)
        for error in errors:
            print("  " + error, file=sys.stderr)
        return 1
    print(f"docs-check: {len(files)} Markdown files OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
