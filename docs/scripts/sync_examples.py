#!/usr/bin/env python3
"""Synchronize documentation example links with the notebooks in ``examples``.

This script mirrors the directory tree under ``examples`` into
``docs/source/examples`` by creating ``.nblink`` files that reference the
original notebooks.  In addition, ``index.rst`` files are generated for every
folder so that the tree can be included in the Sphinx toctree without having to
maintain the structure manually.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
DOCS_EXAMPLES_DIR = REPO_ROOT / "docs" / "source" / "examples"


def _format_title(path: Path) -> str:
    if path == Path("."):
        return "Examples"
    name = path.name.replace("_", " ")
    return name[:1].upper() + name[1:]


def _write_index(directory: Path, rel_path: Path, subdirs: Iterable[str], files: Iterable[str]) -> None:
    title = _format_title(rel_path)
    toctree_entries: List[str] = []
    for subdir in sorted(subdirs):
        toctree_entries.append(f"{subdir}/index")
    for file in sorted(files):
        stem = Path(file).stem
        toctree_entries.append(stem)

    maxdepth = 2 if rel_path == Path(".") else 1
    intro = "Automatically generated overview of example notebooks." if rel_path == Path(".") else ""

    lines = [title, "=" * len(title), ""]
    if intro:
        lines.append(intro)
        lines.append("")
    lines.extend([".. toctree::", f"   :maxdepth: {maxdepth}", ""])
    for entry in toctree_entries:
        lines.append(f"   {entry}")
    content = "\n".join(lines) + "\n"
    (directory / "index.rst").write_text(content, encoding="utf-8")


def _write_notebook_link(directory: Path, notebook: Path) -> None:
    dest = directory / (notebook.stem + ".nblink")
    relative_path = os.path.relpath(notebook, directory)
    data = {"path": relative_path.replace(os.sep, "/")}
    dest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if not EXAMPLES_DIR.exists():
        raise SystemExit(f"Examples directory not found: {EXAMPLES_DIR}")

    if DOCS_EXAMPLES_DIR.exists():
        for path in sorted(DOCS_EXAMPLES_DIR.glob("**/*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    DOCS_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for current_dir, dirnames, filenames in os.walk(EXAMPLES_DIR):
        dirnames[:] = [d for d in dirnames if d != "data"]
        dir_path = Path(current_dir)
        rel_path = dir_path.relative_to(EXAMPLES_DIR)
        dest_dir = DOCS_EXAMPLES_DIR / rel_path
        dest_dir.mkdir(parents=True, exist_ok=True)

        notebooks = [f for f in filenames if f.endswith(".ipynb")]
        for notebook_name in notebooks:
            notebook_path = dir_path / notebook_name
            _write_notebook_link(dest_dir, notebook_path)

        _write_index(dest_dir, rel_path if rel_path.parts else Path("."), dirnames, notebooks)


if __name__ == "__main__":
    main()
