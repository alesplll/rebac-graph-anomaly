"""Running scorers over several seeds and writing the result down.

Section 9 of the spec asks for five fixed seeds per configuration with mean and
standard deviation. A single run of a stochastic estimator on synthetic data is an
anecdote: the seed moves both the dataset and the estimator, and the spread
between seeds is itself a result worth reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from rga.baselines.base import Scorer
from rga.baselines.outliers import IsolationForestScorer, LocalOutlierFactorScorer
from rga.baselines.rules import RuleScorer
from rga.eval.metrics import evaluate_ranking, recall_by_pattern
from rga.features.build import Span, build_candidates
from rga.features.spec import CandidateSet, FeatureGroup
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

#: Queue size the per-pattern breakdown is measured at.
_PATTERN_K = 50


@dataclass(frozen=True)
class ExperimentConfig:
    """One comparison to run."""

    name: str
    dataset: Path
    seeds: tuple[int, ...]
    scorers: tuple[str, ...]
    ks: tuple[int, ...]
    #: Feature-group subsets for the ablation study; None means the full set.
    feature_groups: tuple[tuple[FeatureGroup, ...], ...] | None


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load an experiment recipe from YAML."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw_groups = document.get("feature_groups")
    groups = (
        tuple(tuple(FeatureGroup(name) for name in variant) for variant in raw_groups)
        if raw_groups
        else None
    )
    return ExperimentConfig(
        name=str(document["name"]),
        dataset=Path(document["dataset"]),
        seeds=tuple(int(seed) for seed in document["seeds"]),
        scorers=tuple(str(name) for name in document["scorers"]),
        ks=tuple(int(k) for k in document["ks"]),
        feature_groups=groups,
    )


def build_scorer(name: str, seed: int) -> Scorer:
    """Construct a scorer by the name used in configuration files."""
    if name == "rules":
        return RuleScorer()
    if name == "isolation_forest":
        return IsolationForestScorer(seed=seed)
    if name == "lof":
        return LocalOutlierFactorScorer()
    if name == "gnn":
        from rga.nn.scorer import GnnScorer

        return GnnScorer(seed=seed)
    if name == "gnn_supervised":
        from rga.nn.supervised import SupervisedGnnScorer

        return SupervisedGnnScorer(seed=seed)
    if name == "gnn_forest":
        from rga.nn.hybrid import GnnAugmentedForestScorer

        return GnnAugmentedForestScorer(seed=seed)
    raise KeyError(f"unknown scorer: {name!r}")


def restrict_candidates(
    candidates: CandidateSet, groups: tuple[FeatureGroup, ...]
) -> CandidateSet:
    """The same candidates seen through a narrower feature set.

    This is how a weaker authorization engine is simulated: level 0 keeps the
    structural group only, level 1 adds temporal, level 2 adds provenance.
    """
    return CandidateSet(
        keys=candidates.keys,
        ts=candidates.ts,
        matrix=candidates.matrix.with_groups(groups),
        labels=candidates.labels,
        patterns=candidates.patterns,
        graph=candidates.graph,
    )


def _variant_name(groups: tuple[FeatureGroup, ...] | None) -> str:
    return "all" if groups is None else "+".join(str(group) for group in groups)


@dataclass(frozen=True)
class ExperimentResult:
    """Every measurement from one experiment."""

    config: ExperimentConfig
    rows: tuple[dict[str, object], ...]
    #: Scorer name to pattern name to recall at the reporting queue size.
    per_pattern: dict[str, dict[str, float]]

    def aggregate(self) -> dict[str, dict[str, tuple[float, float]]]:
        """Mean and standard deviation of every metric, per scorer."""
        summary: dict[str, dict[str, tuple[float, float]]] = {}
        for scorer in {str(row["scorer"]) for row in self.rows}:
            own = [row for row in self.rows if row["scorer"] == scorer]
            metrics: dict[str, tuple[float, float]] = {}
            for key, value in own[0].items():
                if not isinstance(value, float):
                    continue
                series = np.array([float(row[key]) for row in own], dtype=np.float64)
                metrics[key] = (float(np.nanmean(series)), float(np.nanstd(series)))
            summary[scorer] = metrics
        return summary


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run every scorer on every seed and collect the numbers."""
    base = load_dataset_config(config.dataset)
    rows: list[dict[str, object]] = []
    pattern_totals: dict[str, dict[str, list[float]]] = {}
    variants = config.feature_groups or (None,)

    for seed in config.seeds:
        dataset = build_dataset(replace(base, seed=seed))
        train = build_candidates(dataset, Span.TRAIN)
        evaluation = build_candidates(dataset, Span.EVAL)

        for groups in variants:
            train_view = train if groups is None else restrict_candidates(train, groups)
            eval_view = (
                evaluation if groups is None else restrict_candidates(evaluation, groups)
            )
            for name in config.scorers:
                scorer = build_scorer(name, seed=seed)
                scorer.fit(train_view)
                scores = scorer.score(eval_view)

                metrics = evaluate_ranking(eval_view.y_true(), scores, ks=config.ks)
                rows.append(
                    {
                        "scorer": name,
                        "seed": seed,
                        "groups": _variant_name(groups),
                        **metrics.as_row(),
                    }
                )

                found = recall_by_pattern(
                    eval_view.y_true(), scores, eval_view.patterns, k=_PATTERN_K
                )
                bucket = pattern_totals.setdefault(name, {})
                for pattern, recall in found.items():
                    bucket.setdefault(pattern, []).append(recall)

    per_pattern = {
        name: {pattern: float(np.mean(values)) for pattern, values in patterns.items()}
        for name, patterns in pattern_totals.items()
    }
    return ExperimentResult(config=config, rows=tuple(rows), per_pattern=per_pattern)


def format_results_table(result: ExperimentResult) -> str:
    """Render the aggregate as a Markdown table, ready to paste into the thesis."""
    aggregate = result.aggregate()
    headline = ("pr_auc", "roc_auc", *(f"precision_at_{k}" for k in result.config.ks))

    lines = [f"### {result.config.name} — {len(result.config.seeds)} seeds", ""]
    lines.append("| scorer | " + " | ".join(headline) + " |")
    lines.append("|---" * (len(headline) + 1) + "|")
    for scorer in sorted(aggregate):
        cells = []
        for metric in headline:
            mean, deviation = aggregate[scorer].get(metric, (float("nan"), float("nan")))
            cells.append(f"{mean:.3f} ± {deviation:.3f}")
        lines.append(f"| {scorer} | " + " | ".join(cells) + " |")

    if result.per_pattern:
        patterns = sorted({p for byname in result.per_pattern.values() for p in byname})
        lines += ["", f"### Recall at {_PATTERN_K}, by pattern", ""]
        lines.append("| scorer | " + " | ".join(patterns) + " |")
        lines.append("|---" * (len(patterns) + 1) + "|")
        for scorer in sorted(result.per_pattern):
            cells = [f"{result.per_pattern[scorer].get(p, 0.0):.2f}" for p in patterns]
            lines.append(f"| {scorer} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


#: The five patterns the supervised baseline is trained on.
KNOWN_PATTERNS = (
    "self_grant_admin",
    "privileged_group_join",
    "grant_burst",
    "hierarchy_bypass",
    "cross_department",
)
#: The three it never sees. How a model does here is what section 9 is after.
HIDDEN_PATTERNS = ("dormant_awakening", "shadow_group", "delegation_cascade")


def format_hidden_pattern_table(result: ExperimentResult) -> str:
    """Recall on trained patterns against recall on unseen ones.

    A model that learned the nature of an anomaly keeps its recall on patterns it
    has never seen. A model that learned the generator loses it, and the gap column
    measures exactly that.
    """
    lines = [
        f"### Recall at {_PATTERN_K}: known patterns against hidden ones",
        "",
        "| scorer | known | hidden | gap |",
        "|---|---|---|---|",
    ]
    for scorer in sorted(result.per_pattern):
        found = result.per_pattern[scorer]
        known = np.nanmean([found.get(name, 0.0) for name in KNOWN_PATTERNS])
        hidden = np.nanmean([found.get(name, 0.0) for name in HIDDEN_PATTERNS])
        lines.append(f"| {scorer} | {known:.2f} | {hidden:.2f} | {known - hidden:+.2f} |")

    lines += [
        "",
        "Известными считаются паттерны 1-5 из раздела 5.3, на которых обучается",
        "супервизорный вариант; скрытыми — паттерны 6-8, которых он не видел.",
        "Самообучаемая модель не видит меток вообще, поэтому для неё все восемь",
        "одинаково незнакомы, и разрыв у неё должен быть близок к нулю.",
    ]
    return "\n".join(lines)


def format_ablation_table(result: ExperimentResult) -> str:
    """Quality per feature-group variant.

    Doubles as the integration guide from section 12: each row is what an engine at
    that capability level can deliver.
    """
    variants = sorted({str(row["groups"]) for row in result.rows})
    scorers = sorted({str(row["scorer"]) for row in result.rows})

    lines = [f"### {result.config.name} — feature groups", ""]
    lines.append("| feature groups | " + " | ".join(scorers) + " |")
    lines.append("|---" * (len(scorers) + 1) + "|")
    for variant in variants:
        cells = []
        for scorer in scorers:
            series = [
                float(row["pr_auc"])
                for row in result.rows
                if row["groups"] == variant and row["scorer"] == scorer
            ]
            cells.append(
                f"{np.nanmean(series):.3f} ± {np.nanstd(series):.3f}" if series else "—"
            )
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")

    lines += ["", "pr_auc, mean ± standard deviation over seeds."]
    return "\n".join(lines)


def save_results(path: Path, result: ExperimentResult) -> None:
    """Write the raw rows and the rendered table side by side."""
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "name": result.config.name,
            "dataset": str(result.config.dataset),
            "seeds": list(result.config.seeds),
            "scorers": list(result.config.scorers),
            "ks": list(result.config.ks),
        },
        "rows": [dict(row) for row in result.rows],
        "per_pattern": result.per_pattern,
        "aggregate": {
            scorer: {metric: list(pair) for metric, pair in metrics.items()}
            for scorer, metrics in result.aggregate().items()
        },
    }
    (path / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    table = format_results_table(result)
    if {str(row["groups"]) for row in result.rows} != {"all"}:
        table += "\n\n" + format_ablation_table(result)
    if result.per_pattern:
        table += "\n\n" + format_hidden_pattern_table(result)
    (path / "results.md").write_text(table + "\n", encoding="utf-8")
