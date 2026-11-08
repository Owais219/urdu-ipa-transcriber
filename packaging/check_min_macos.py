"""Fail if any compiled file in the app needs a newer macOS than the app promises.

    python packaging/check_min_macos.py "dist/Urdu IPA.app" 12.0
"""
import subprocess
import sys
from pathlib import Path


def min_os(path: Path):
    try:
        out = subprocess.run(["otool", "-l", str(path)], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    lines = out.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("minos "):                       # LC_BUILD_VERSION
            return s.split()[1]
        if s == "cmd LC_VERSION_MIN_MACOSX":
            for nxt in lines[i + 1:i + 4]:
                if nxt.strip().startswith("version "):
                    return nxt.strip().split()[1]
    return None


def key(v):
    return tuple(int(x) for x in v.split(".")[:2])


def main(app, limit):
    bad, n = [], 0
    for p in Path(app).rglob("*"):
        if p.is_symlink() or not p.is_file():
            continue
        with open(p, "rb") as f:
            magic = f.read(4)
        if magic not in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xce\xfa\xed\xfe"):
            continue
        n += 1
        v = min_os(p)
        if v and key(v) > key(limit):
            bad.append((v, str(p)))
    print(f"checked {n} Mach-O files against macOS {limit}")
    for v, p in sorted(bad, reverse=True):
        print(f"  needs macOS {v}: {p}")
    if bad:
        sys.exit(f"{len(bad)} files need a newer macOS than {limit}")
    print("all compatible")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "12.0")
