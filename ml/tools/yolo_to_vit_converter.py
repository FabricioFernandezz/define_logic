"""Ejecuta scripts/yolo_to_vit_converter.py desde estructura organizada.

Uso:
    python -m ml.tools.yolo_to_vit_converter
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2] / "scripts" / "yolo_to_vit_converter.py"
    runpy.run_path(str(target), run_name="__main__")
