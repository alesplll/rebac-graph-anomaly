"""Restricting the feature set, for the ablation study and the integration guide."""

from pathlib import Path

from rga.eval.experiment import (
    ExperimentConfig,
    format_ablation_table,
    restrict_candidates,
    run_experiment,
)
from rga.features.build import Span, build_candidates
from rga.features.spec import FeatureGroup
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))

STRUCTURAL_ONLY = (FeatureGroup.STRUCTURAL,)
WITH_TIME = (FeatureGroup.STRUCTURAL, FeatureGroup.TEMPORAL)
EVERYTHING = (FeatureGroup.STRUCTURAL, FeatureGroup.TEMPORAL, FeatureGroup.PROVENANCE)


def _candidates():
    return build_candidates(build_dataset(CONFIG), Span.EVAL)


def _config(groups):
    return ExperimentConfig(
        name="ablation-test",
        dataset=Path("configs/generator/small.yaml"),
        seeds=(1,),
        scorers=("isolation_forest",),
        ks=(50,),
        feature_groups=groups,
    )


def test_restriction_narrows_the_matrix_but_keeps_the_rows() -> None:
    candidates = _candidates()
    reduced = restrict_candidates(candidates, STRUCTURAL_ONLY)

    assert reduced.n_candidates == candidates.n_candidates
    assert reduced.matrix.n_features < candidates.matrix.n_features
    assert set(reduced.matrix.block.groups) == {FeatureGroup.STRUCTURAL}


def test_restriction_preserves_labels_and_keys() -> None:
    candidates = _candidates()
    reduced = restrict_candidates(candidates, WITH_TIME)

    assert reduced.keys == candidates.keys
    assert reduced.patterns == candidates.patterns
    assert (reduced.labels == candidates.labels).all()


def test_capability_levels_are_nested() -> None:
    # Level 0 sees structural only, level 1 adds time, level 2 adds provenance.
    candidates = _candidates()
    widths = [
        restrict_candidates(candidates, groups).matrix.n_features
        for groups in (STRUCTURAL_ONLY, WITH_TIME, EVERYTHING)
    ]
    assert widths[0] < widths[1] < widths[2]


def test_experiment_runs_every_variant() -> None:
    result = run_experiment(_config((STRUCTURAL_ONLY, EVERYTHING)))
    assert len(result.rows) == 2
    assert {row["groups"] for row in result.rows} == {
        "structural",
        "structural+temporal+provenance",
    }


def test_ablation_table_names_the_variants() -> None:
    table = format_ablation_table(run_experiment(_config((STRUCTURAL_ONLY, EVERYTHING))))
    assert "structural" in table
    assert "pr_auc" in table


def test_full_run_without_variants_labels_rows_as_full() -> None:
    config = ExperimentConfig(
        name="plain",
        dataset=Path("configs/generator/small.yaml"),
        seeds=(1,),
        scorers=("rules",),
        ks=(50,),
        feature_groups=None,
    )
    assert {row["groups"] for row in run_experiment(config).rows} == {"all"}
