"""Run pytest with the project interpreter and write the output to a file.

A thin wrapper around pytest that works around command-line quoting issues
in some shells.  Usage::

    python scripts/run_tests.py
    python scripts/run_tests.py tests/test_losses.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = sys.argv[1:]
    if not args:
        args = ["tests", "-q", "--no-header", "-p", "no:cacheprovider"]

    command = [sys.executable, "-m", "pytest", *args]
    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(ROOT))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
