"""Make every bundled copy of the Intel OpenMP runtime (libiomp5.dylib) point to ONE file.

PyTorch (torch/lib and functorch/.dylibs) and CTranslate2 (ctranslate2/.dylibs) each ship
their own libiomp5.dylib on Intel Macs. If two copies are loaded into one process the
OpenMP runtime aborts or crashes. macOS identifies a loaded library by its real path, so
replacing the extra copies with symlinks to a single file makes it load exactly once.
The newest copy (CTranslate2's, if present) is kept; libiomp5 is backward compatible.

    python packaging/dedupe_openmp.py <site-packages dir or .app bundle>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

NAME = "libiomp5.dylib"


def dedupe(root: Path) -> Path | None:
    copies = sorted({p.resolve() for p in root.rglob(NAME)})
    if not copies:
        print(f"no {NAME} under {root}")
        return None
    canonical = next((p for p in copies if "ctranslate2" in p.parts), None) or max(copies, key=lambda p: p.stat().st_mtime)
    for p in root.rglob(NAME):
        if p.is_symlink() and p.resolve() == canonical:
            continue
        if p.resolve() == canonical:
            continue
        rel = os.path.relpath(canonical, p.parent)
        p.unlink()
        p.symlink_to(rel)
        print(f"{p} -> {rel}")
    real = {p.resolve() for p in root.rglob(NAME)}
    if len(real) != 1:
        sys.exit(f"still {len(real)} distinct copies: {real}")
    print(f"one OpenMP runtime: {canonical}")
    return canonical


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        dedupe(Path(arg))
