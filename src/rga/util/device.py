"""Torch device selection.

Torch is imported lazily: the data and generator layers of this project run
without it, and the laptop used for development installs the CPU build only.
"""

from __future__ import annotations

# Minimum compute capability of the training GPU (RTX 5060 Ti, Blackwell).
# Wheels built against CUDA 12.1 and earlier do not contain sm_120 kernels and
# fail at the first kernel launch rather than at import time.
_TARGET_CAPABILITY = (12, 0)


def select_device(prefer: str = "auto") -> str:
    """Return the torch device string to use.

    `auto` picks CUDA when it is usable, `cuda` degrades to CPU rather than
    raising so the same command line works on both machines, and `cpu` forces
    the CPU path.
    """
    if prefer not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"unknown device preference: {prefer!r}")
    if prefer == "cpu":
        return "cpu"

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device() -> dict[str, object]:
    """Collect everything needed to diagnose a broken CUDA install."""
    import torch

    available = bool(torch.cuda.is_available())
    capability: tuple[int, int] | None = None
    name: str | None = None
    if available:
        capability = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)

    return {
        "torch": torch.__version__,
        "cuda_available": available,
        "device": select_device("auto"),
        "capability": capability,
        "name": name,
        "supported": capability is not None and capability >= _TARGET_CAPABILITY,
    }
