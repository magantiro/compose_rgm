"""Torch device selection for CPU, CUDA, and Apple Metal (MPS)."""

from __future__ import annotations

import torch


def resolve_torch_device(requested: str = "auto") -> torch.device:
    name = requested.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is unavailable to this Python process; "
            f"mps_built={torch.backends.mps.is_built()}"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be one of auto, cpu, mps, or cuda")
    return torch.device(name)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
