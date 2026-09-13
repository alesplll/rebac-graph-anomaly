"""Verify that the installed torch build actually runs on this machine's GPU.

Run on the training PC after `uv sync --extra gpu`:

    uv run python scripts/gpu_smoke.py

A build without sm_120 kernels imports cleanly and reports CUDA as available,
then fails on the first real kernel launch. The matmul below is that launch.
"""

from __future__ import annotations

import sys

from rga.util.device import describe_device, select_device


def main() -> int:
    info = describe_device()
    for key, value in info.items():
        print(f"{key:16}: {value}")

    device = select_device("auto")
    if device == "cpu":
        print("\nNo CUDA device. This is expected on the development laptop.")
        return 0

    import torch

    print("\nRunning a matmul on the device...")
    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)
    c = a @ b
    torch.cuda.synchronize()
    print(f"OK: result sum = {float(c.sum()):.4f}")

    if not info["supported"]:
        print("\nWARNING: compute capability is below the expected sm_120.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
