"""Where tensors live and where randomness comes from.

Both are decided once, here, and passed down. Development runs on a CPU laptop and
training on a Blackwell card under Windows, so no module below may assume an
accelerator exists, and reproducibility rules out the implicit global RNG.
"""

from __future__ import annotations

import torch


def select_device(*, prefer_cuda: bool = True) -> torch.device:
    """The device to run on: CUDA when present and wanted, CPU otherwise."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_torch(seed: int) -> None:
    """Seed the global torch RNG, which parameter initialisation reads."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_generator(seed: int, device: torch.device) -> torch.Generator:
    """An explicitly seeded generator for every draw made during training."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
