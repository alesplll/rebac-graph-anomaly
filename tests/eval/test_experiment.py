"""Running baselines over seeds and writing the result down."""

import json
from pathlib import Path

import pytest

from rga.eval.experiment import (
    ExperimentConfig,
    build_scorer,
    format_results_table,
    load_experiment_config,
    run_experiment,
    save_results,
)


def _small(seeds=(1, 2)) -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
        dataset=Path("configs/generator/small.yaml"),
        seeds=seeds,
        scorers=("rules", "isolation_forest"),
        ks=(20, 50),
        feature_groups=None,
    )


def test_shipped_config_loads() -> None:
    config = load_experiment_config(Path("configs/experiments/baselines.yaml"))
    assert config.seeds
    assert config.scorers
    assert config.dataset.exists()


def test_unknown_scorer_is_rejected_by_name() -> None:
    with pytest.raises(KeyError, match="unknown scorer"):
        build_scorer("magic", seed=0)


def test_every_declared_scorer_can_be_built() -> None:
    for name in ("rules", "isolation_forest", "lof"):
        assert build_scorer(name, seed=0).name == name


def test_run_produces_one_row_per_scorer_and_seed() -> None:
    result = run_experiment(_small())
    assert len(result.rows) == 2 * 2
    assert {row["scorer"] for row in result.rows} == {"rules", "isolation_forest"}
    assert {row["seed"] for row in result.rows} == {1, 2}


def test_rows_carry_the_metrics() -> None:
    result = run_experiment(_small(seeds=(1,)))
    row = result.rows[0]
    assert "pr_auc" in row
    assert "precision_at_20" in row
    assert "lift_at_50" in row


def test_aggregate_reports_mean_and_deviation() -> None:
    aggregate = run_experiment(_small()).aggregate()
    mean, deviation = aggregate["rules"]["pr_auc"]
    assert 0.0 <= mean <= 1.0
    assert deviation >= 0.0


def test_per_pattern_recall_is_collected() -> None:
    result = run_experiment(_small(seeds=(1,)))
    assert result.per_pattern
    for scorer_name, by_pattern in result.per_pattern.items():
        assert scorer_name in {"rules", "isolation_forest"}
        assert all(0.0 <= value <= 1.0 for value in by_pattern.values())


def test_table_mentions_every_scorer() -> None:
    table = format_results_table(run_experiment(_small(seeds=(1,))))
    assert "rules" in table
    assert "isolation_forest" in table
    assert "pr_auc" in table


def test_results_are_saved_as_json_and_markdown(tmp_path: Path) -> None:
    result = run_experiment(_small(seeds=(1,)))
    save_results(tmp_path / "run", result)

    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == [
        "results.json",
        "results.md",
    ]
    payload = json.loads((tmp_path / "run" / "results.json").read_text(encoding="utf-8"))
    assert payload["config"]["name"] == "test"
    assert payload["rows"]
