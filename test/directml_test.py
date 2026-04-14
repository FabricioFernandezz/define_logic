import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gpu_usage_test import check_device_usage


def test_directml():
    assert check_device_usage(verbose=False), (
        "DirectML no esta disponible como dispositivo principal o no se esta usando GPU"
    )


if __name__ == "__main__":
    try:
        success = check_device_usage(verbose=True)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[ADVERTENCIA] Prueba cancelada por usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
