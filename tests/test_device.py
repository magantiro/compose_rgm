from __future__ import annotations

import torch

from compose_v4.model.device import resolve_torch_device


def test_auto_device_resolves_to_an_available_backend() -> None:
    device = resolve_torch_device("auto")
    assert device.type in {"cpu", "cuda", "mps"}
    if device.type == "mps":
        assert torch.backends.mps.is_available()
    if device.type == "cuda":
        assert torch.cuda.is_available()


def test_cpu_device_is_always_explicitly_available() -> None:
    assert resolve_torch_device("cpu") == torch.device("cpu")
