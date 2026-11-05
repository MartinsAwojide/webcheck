#!/usr/bin/env python3
"""Runs the demo launcher (webcheck.launcher) in the board's demo folder, with the board's plain python3 (no uv): the
package is in pkg/ next to this file, as for demo_driver.py --pkg pkg.

    python3 webcheck_demo.py start [--clear-db] [SLICES|max] [--open] [--frames DIR] | stop | status
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pkg"))
os.chdir(HERE)

from webcheck.launcher import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
