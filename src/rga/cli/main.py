"""Command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.generator.stats import dataset_stats, format_stats
from rga.io.dataset_io import load_dataset, save_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rga", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate a synthetic dataset")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--out", type=Path, required=True)

    stats = commands.add_parser("stats", help="summarise a dataset")
    stats.add_argument("--dataset", type=Path, required=True)

    evaluate = commands.add_parser("evaluate", help="run baselines and write results")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)

    train = commands.add_parser("train", help="fit a scorer and save it as an artefact")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line. Returns the process exit code."""
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as exit_signal:
        return int(exit_signal.code or 0)

    if arguments.command == "generate":
        dataset = build_dataset(load_dataset_config(arguments.config))
        save_dataset(arguments.out, dataset)
        print(format_stats(dataset_stats(dataset)))
        return 0

    if arguments.command == "stats":
        print(format_stats(dataset_stats(load_dataset(arguments.dataset))))
        return 0

    if arguments.command == "train":
        import yaml

        from rga.artifacts import save_scorer
        from rga.eval.experiment import build_scorer
        from rga.features.build import Span, build_candidates
        from rga.nn.config import ModelConfig

        recipe = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
        name = str(recipe["scorer"])
        if name not in {"gnn", "gnn_supervised"}:
            print(f"no artefact format for scorer {name!r}")
            return 2

        seed = int(recipe.get("seed", 0))
        dataset_config = load_dataset_config(Path(recipe["dataset"]))
        train_set = build_candidates(build_dataset(dataset_config), Span.TRAIN)

        scorer = build_scorer(name, seed=seed)
        overrides = recipe.get("model") or {}
        if overrides:
            scorer = type(scorer)(seed=seed, config=ModelConfig(**overrides))
        scorer.fit(train_set)
        save_scorer(arguments.out, scorer, dataset=dataset_config.name)
        print(f"wrote {arguments.out}")
        return 0

    if arguments.command == "evaluate":
        from rga.eval.experiment import (
            format_results_table,
            load_experiment_config,
            run_experiment,
            save_results,
        )

        result = run_experiment(load_experiment_config(arguments.config))
        save_results(arguments.out, result)
        print(format_results_table(result))
        return 0

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
