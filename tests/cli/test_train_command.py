"""Training a shipped artefact from the command line."""

from pathlib import Path

from rga.artifacts import read_manifest
from rga.cli.main import main


def _recipe(tmp_path: Path) -> Path:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "dataset: configs/generator/small-history.yaml\n"
        "scorer: gnn_supervised\n"
        "seed: 5\n"
        "model:\n"
        "  hidden_dim: 16\n"
        "  num_layers: 2\n"
        "  epochs: 3\n"
        "  patience: 3\n",
        encoding="utf-8",
        newline="\n",
    )
    return recipe


def test_training_writes_a_loadable_artefact(tmp_path) -> None:
    out = tmp_path / "artifact"

    code = main(["train", "--config", str(_recipe(tmp_path)), "--out", str(out)])

    assert code == 0
    assert (out / "model.pt").exists()
    assert (out / "state.npz").exists()
    manifest = read_manifest(out)
    assert manifest["scorer"] == "gnn_supervised"
    assert manifest["seed"] == 5


def test_an_unknown_scorer_is_refused(tmp_path) -> None:
    recipe = tmp_path / "bad.yaml"
    recipe.write_text(
        "dataset: configs/generator/small-history.yaml\nscorer: nonsense\nseed: 1\n",
        encoding="utf-8",
        newline="\n",
    )

    assert main(["train", "--config", str(recipe), "--out", str(tmp_path / "x")]) == 2
