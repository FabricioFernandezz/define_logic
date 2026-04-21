"""
Device Manager para gestionar GPU DirectML en PyTorch
Detecta automáticamente si DirectML está disponible y lo configura
"""
import torch
import logging
from typing import Literal

logger = logging.getLogger(__name__)


class DeviceManager:
    """Gestor centralizado de dispositivos de GPU/CPU"""
    
    @staticmethod
    def get_device(device_config: str = 'directml') -> torch.device:
        """
        Obtiene el dispositivo de PyTorch basado en la configuración
        
        Args:
            device_config: 'directml' o 'cpu'
            
        Returns:
            torch.device: Dispositivo configurado
        """
        
        # Intentar DirectML
        if device_config.lower() in ['directml', 'auto']:
            try:
                import torch_directml
                device = torch_directml.device()
                # Verificar que funciona
                _ = torch.zeros(1, device=device)
                logger.info("[OK] DirectML GPU detectada y configurada")
                return device
            except Exception as e:
                logger.warning(f"DirectML no disponible: {e}")
        
        # Fallback a CPU
        logger.info("Usando CPU")
        return torch.device('cpu')
    
    @staticmethod
    def get_device_strict(force_gpu: bool = True) -> torch.device:
        """
        Obtiene dispositivo con validación ESTRICTA
        
        Args:
            force_gpu: Si True, FALLA si no hay GPU disponible
            
        Returns:
            torch.device: Dispositivo GPU
            
        Raises:
            RuntimeError: Si force_gpu=True y DirectML no está disponible
        """
        try:
            import torch_directml
            device = torch_directml.device()
            # Verificar que funciona
            _ = torch.zeros(1, device=device)
            logger.info("[OK] DirectML GPU disponible y verificado")
            return device
        except Exception as e:
            if force_gpu:
                raise RuntimeError(
                    f"\n{'='*70}\n"
                    f"[FATAL ERROR] DirectML GPU NO DISPONIBLE\n"
                    f"{'='*70}\n"
                    f"Error: {e}\n"
                    f"\nEl entrenamiento requiere GPU DirectML obligatoriamente.\n"
                    f"CPU NO está permitido en este proyecto.\n"
                    f"\nVerifica:\n"
                    f"  1. ¿torch-directml está instalado?\n"
                    f"     pip install torch-directml\n"
                    f"  2. ¿Tienes GPU compatible (AMD/Intel)?\n"
                    f"  3. ¿Los drivers están actualizados?\n"
                    f"{'='*70}\n"
                ) from e
            logger.warning(f"DirectML no disponible, usando CPU: {e}")
            return torch.device('cpu')
    
    @staticmethod
    def print_device_info() -> None:
        """Imprime información sobre los dispositivos disponibles"""
        print("=" * 60)
        print("INFORMACIÓN DE DISPOSITIVOS DISPONIBLES")
        print("=" * 60)
        
        print(f"PyTorch version: {torch.__version__}")
        
        # DirectML
        try:
            import torch_directml
            device = torch_directml.device()
            _ = torch.zeros(1, device=device)
            print("[OK] DirectML: Disponible")
        except Exception as e:
            print(f"[ERROR] DirectML: No disponible ({str(e)[:50]})")
        
        print("[OK] CPU: Disponible")
        print("=" * 60)


# Funciones de conveniencia
def get_device(device_config: str = 'directml') -> torch.device:
    """Obtiene el dispositivo recomendado"""
    return DeviceManager.get_device(device_config)


def get_device_strict(force_gpu: bool = True) -> torch.device:
    """Obtiene dispositivo con validación ESTRICTA - FALLA si no hay GPU"""
    return DeviceManager.get_device_strict(force_gpu)


def print_device_info() -> None:
    """Imprime información sobre dispositivos"""
    DeviceManager.print_device_info()
