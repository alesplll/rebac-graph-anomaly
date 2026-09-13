"""Device description must work on a machine with no CUDA at all."""

import pytest

from rga.util.device import describe_device, select_device

torch = pytest.importorskip("torch")


def test_describe_device_reports_expected_keys() -> None:
    info = describe_device()
    assert set(info) == {"torch", "cuda_available", "device", "capability", "name", "supported"}
    assert isinstance(info["torch"], str)
    assert isinstance(info["cuda_available"], bool)


def test_select_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto") == "cpu"
    assert select_device("cuda") == "cpu"


def test_select_device_rejects_unknown_preference() -> None:
    with pytest.raises(ValueError, match="unknown device preference"):
        select_device("tpu")
