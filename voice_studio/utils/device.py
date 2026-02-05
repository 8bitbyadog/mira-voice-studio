"""
Device selection for Apple Silicon (M1/M2/M3) with MPS acceleration.

This module handles GPU device selection, preferring Metal Performance Shaders (MPS)
on Apple Silicon Macs, with automatic fallback to CPU.
"""

import torch
from typing import Tuple


def get_device() -> torch.device:
    """
    Get the optimal compute device for the current system.

    Returns MPS (Metal Performance Shaders) on Apple Silicon Macs,
    falls back to CPU if MPS is not available.

    Returns:
        torch.device: The selected compute device (mps or cpu).
    """
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return torch.device("mps")
    return torch.device("cpu")


def get_device_info() -> dict:
    """
    Get detailed information about the available compute devices.

    Returns:
        dict: Device information including:
            - device: The selected device name
            - mps_available: Whether MPS is available
            - mps_built: Whether MPS was built into PyTorch
            - cuda_available: Whether CUDA is available (should be False on Mac)
    """
    device = get_device()

    return {
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built() if torch.backends.mps.is_available() else False,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
    }


def check_device_memory() -> Tuple[bool, str]:
    """
    Check if the device has sufficient memory for TTS operations.

    Returns:
        Tuple[bool, str]: (is_sufficient, message)
    """
    device = get_device()

    if device.type == "mps":
        # MPS doesn't have a direct memory query API like CUDA
        # We assume Apple Silicon with MPS has sufficient unified memory
        return True, "MPS device detected. Using Apple Silicon unified memory."

    # CPU fallback
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        if available_gb < 4:
            return False, f"Low memory: {available_gb:.1f}GB available. Recommend 8GB+."
        return True, f"CPU mode with {available_gb:.1f}GB available memory."
    except ImportError:
        return True, "CPU mode. Memory check unavailable (psutil not installed)."


def move_to_device(tensor_or_model, device: torch.device = None):
    """
    Move a tensor or model to the specified device.

    Args:
        tensor_or_model: A PyTorch tensor or model.
        device: Target device. If None, uses get_device().

    Returns:
        The tensor or model on the target device.
    """
    if device is None:
        device = get_device()
    return tensor_or_model.to(device)
