import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from utils.device_manager import get_device


def _check_directml() -> tuple[bool, str, str | None]:
    try:
        import torch_directml

        dml_device = torch_directml.device()
        _ = torch.tensor([1.0], device=dml_device)
        return True, str(dml_device), None
    except Exception as err:
        return False, "N/A", str(err)


def _is_gpu_device(device_value: object) -> bool:
    device_text = str(device_value).lower()
    return any(token in device_text for token in ("privateuseone", "cuda", "xpu", "mps"))


def check_device_usage(verbose: bool = True) -> bool:
    vit_config = getattr(config, "ViT_TRAINING_CONFIG", {})
    yolo_config = getattr(config, "YOLO_TRAINING_CONFIG", {})
    default_device = getattr(config, "DEVICE", "cpu")

    configured_vit = str(vit_config.get("device", default_device)).lower()
    configured_yolo = str(yolo_config.get("device", default_device)).lower()
    configured_main = configured_vit if configured_vit else str(default_device).lower()

    directml_available, directml_device, directml_error = _check_directml()
    effective_device = get_device(configured_main)
    using_gpu = _is_gpu_device(effective_device)

    configured_for_gpu = configured_main != "cpu"
    ok = directml_available and using_gpu and configured_for_gpu

    if verbose:
        print("=" * 60)
        print("TEST RAPIDO: DIRECTML PRINCIPAL + USO DE GPU")
        print("=" * 60)
        print(f"PyTorch: {torch.__version__}")
        print(f"Config ViT: {configured_vit}")
        print(f"Config YOLO: {configured_yolo}")
        print(f"DirectML disponible: {directml_available}")
        if directml_available:
            print(f"Dispositivo DirectML: {directml_device}")
        else:
            print(f"Error DirectML: {directml_error}")
        print(f"Dispositivo efectivo: {effective_device}")
        print(f"Usando GPU: {using_gpu}")
        print(f"DirectML configurado como principal: {configured_for_gpu}")
        print(f"RESULTADO: {'OK' if ok else 'FAIL'}")
        print("=" * 60)

    return ok


if __name__ == "__main__":
    sys.exit(0 if check_device_usage(verbose=True) else 1)