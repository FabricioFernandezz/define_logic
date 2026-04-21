"""Entrypoint organizado para entrenamiento ViT completo.

Uso:
    python -m ml.training.train_vit_epp
"""

from __future__ import annotations

from models.train_vit_epp import main


if __name__ == "__main__":
    main()
