"""A dataset directory round-trips exactly."""

from pathlib import Path

from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.io.dataset_io import load_dataset, save_dataset


def test_dataset_round_trips(tmp_path: Path) -> None:
    original = build_dataset(load_dataset_config(Path("configs/generator/small.yaml")))
    save_dataset(tmp_path / "small", original)
    restored = load_dataset(tmp_path / "small")

    assert restored.events == original.events
    assert restored.labels == original.labels
    assert restored.split_ts == original.split_ts
    assert restored.window_end == original.window_end
    assert restored.config == original.config


def test_saved_layout_is_the_expected_three_files(tmp_path: Path) -> None:
    dataset = build_dataset(load_dataset_config(Path("configs/generator/small.yaml")))
    save_dataset(tmp_path / "small", dataset)
    assert sorted(path.name for path in (tmp_path / "small").iterdir()) == [
        "events.jsonl",
        "labels.jsonl",
        "meta.json",
    ]
