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

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
