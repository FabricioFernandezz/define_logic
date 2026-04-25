
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from .app import app
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from back.app import app


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
