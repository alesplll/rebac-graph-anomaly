"""The command line runs end to end."""

from pathlib import Path

from rga.cli.main import main


def test_generate_then_stats(tmp_path: Path, capsys) -> None:
    out = tmp_path / "small"
    assert main(["generate", "--config", "configs/generator/small.yaml", "--out", str(out)]) == 0
    assert (out / "events.jsonl").exists()

    assert main(["stats", "--dataset", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "train_nodes" in printed
    assert "anomalies" in printed


def test_unknown_command_fails_cleanly(capsys) -> None:
    assert main(["nonsense"]) == 2


def test_evaluate_runs_and_writes_results(tmp_path: Path, capsys) -> None:
    # The smoke config, not the real one: a five-seed three-scorer comparison
    # does not belong in a unit test run on every commit.
    out = tmp_path / "run"
    code = main(["evaluate", "--config", "configs/experiments/smoke.yaml", "--out", str(out)])
    assert code == 0
    assert (out / "results.md").exists()
    assert "pr_auc" in capsys.readouterr().out
