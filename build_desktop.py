"""Build the Hypotaxis desktop application with PyInstaller."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    command = [sys.executable, "-m", "PyInstaller", "--clean", "hypotaxis.spec"]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
