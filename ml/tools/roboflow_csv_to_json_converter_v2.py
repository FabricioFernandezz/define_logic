"""Ejecuta scripts/roboflow_csv_to_json_converter_v2.py desde estructura organizada.

Uso:
    python -m ml.tools.roboflow_csv_to_json_converter_v2
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2] / "scripts" / "roboflow_csv_to_json_converter_v2.py"
    runpy.run_path(str(target), run_name="__main__")
