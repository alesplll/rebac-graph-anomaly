"""The table that carries the module's main result."""

from pathlib import Path

from rga.eval.experiment import (
    ExperimentConfig,
    ExperimentResult,
    format_hidden_pattern_table,
)


def _result() -> ExperimentResult:
    config = ExperimentConfig(
        name="demo",
        dataset=Path("configs/generator/small-history.yaml"),
        seeds=(1,),
        scorers=("gnn", "gnn_supervised"),
        ks=(50,),
        feature_groups=None,
    )
    return ExperimentResult(
        config=config,
        rows=(),
        per_pattern={
            "gnn": {
                "self_grant_admin": 1.0,
                "privileged_group_join": 0.8,
                "grant_burst": 0.9,
                "hierarchy_bypass": 1.0,
                "cross_department": 0.8,
                "dormant_awakening": 0.9,
                "shadow_group": 0.9,
                "delegation_cascade": 0.9,
            },
            "gnn_supervised": {
                "self_grant_admin": 1.0,
                "privileged_group_join": 1.0,
                "grant_burst": 1.0,
                "hierarchy_bypass": 1.0,
                "cross_department": 1.0,
                "dormant_awakening": 0.1,
                "shadow_group": 0.2,
                "delegation_cascade": 0.0,
            },
        },
    )


def test_the_table_names_both_groups() -> None:
    table = format_hidden_pattern_table(_result())

    assert "known" in table
    assert "hidden" in table


def test_a_model_that_generalises_shows_a_small_gap() -> None:
    table = format_hidden_pattern_table(_result())
    row = next(line for line in table.splitlines() if line.startswith("| gnn |"))

    assert "0.90" in row
    assert "+0.00" in row


def test_a_model_that_memorised_shows_a_large_gap() -> None:
    table = format_hidden_pattern_table(_result())
    row = next(line for line in table.splitlines() if line.startswith("| gnn_supervised |"))

    assert "1.00" in row
    assert "0.10" in row
    assert "+0.90" in row
