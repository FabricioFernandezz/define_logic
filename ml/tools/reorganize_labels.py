"""Ejecuta scripts/reorganize_labels.py desde estructura organizada.

Uso:
    python -m ml.tools.reorganize_labels
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2] / "scripts" / "reorganize_labels.py"
    runpy.run_path(str(target), run_name="__main__")
