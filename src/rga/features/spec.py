"""Feature naming, grouping and the containers that carry them.

Every feature belongs to a group. That is not bookkeeping: masking a group is how
the ablation study is run, and the same mechanism tells an integrator what a
weaker authorization engine costs them. A feature the source cannot supply is
marked unobserved rather than silently set to zero, because zero is a legitimate
value for most of these and a model cannot tell the two apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class FeatureGroup(StrEnum):
    """Which capability level a feature needs from the source."""

    #: Available from a bare snapshot of relation tuples.
    STRUCTURAL = "structural"
    #: Needs creation timestamps.
    TEMPORAL = "temporal"
    #: Needs the initiator of each change.
    PROVENANCE = "provenance"


@dataclass(frozen=True)
class FeatureBlock:
    """The names and groups of a contiguous run of features."""

    names: tuple[str, ...]
    groups: tuple[FeatureGroup, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.groups):
            raise ValueError("names and groups must have the same length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("duplicate feature name in block")

    def __len__(self) -> int:
        return len(self.names)

    def prefixed(self, prefix: str) -> FeatureBlock:
        """The same block with every name prefixed, for reuse on both endpoints."""
        return FeatureBlock(
            names=tuple(f"{prefix}{name}" for name in self.names), groups=self.groups
        )

    @classmethod
    def concat(cls, *blocks: FeatureBlock) -> FeatureBlock:
        """Join blocks end to end."""
        names: tuple[str, ...] = ()
        groups: tuple[FeatureGroup, ...] = ()
        for block in blocks:
            names += block.names
            groups += block.groups
        return cls(names=names, groups=groups)


@dataclass(frozen=True)
class FeatureMatrix:
    """Feature values and their observability."""

    values: np.ndarray
    mask: np.ndarray
    block: FeatureBlock

    def __post_init__(self) -> None:
        if self.values.shape[1] != len(self.block):
            raise ValueError(
                f"values have {self.values.shape[1]} columns "
                f"but the block names {len(self.block)} features"
            )
        if self.mask.shape != self.values.shape:
            raise ValueError(f"mask shape {self.mask.shape} differs from {self.values.shape}")

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    def group_indices(self, group: FeatureGroup) -> np.ndarray:
        """Column indices belonging to one group."""
        return np.array(
            [index for index, own in enumerate(self.block.groups) if own is group], dtype=np.int64
        )

    def column(self, name: str) -> np.ndarray:
        """One feature's values across all rows, by name."""
        return self.values[:, self._position(name)]

    def observed(self, name: str) -> np.ndarray:
        """Whether one feature was observed, across all rows, by name."""
        return self.mask[:, self._position(name)]

    def _position(self, name: str) -> int:
        try:
            return self.block.names.index(name)
        except ValueError:
            raise KeyError(f"no such feature: {name!r}") from None

    def with_groups(self, groups: tuple[FeatureGroup, ...]) -> FeatureMatrix:
        """A matrix restricted to the named groups, for the ablation study."""
        wanted = set(groups)
        columns = np.array(
            [index for index, own in enumerate(self.block.groups) if own in wanted],
            dtype=np.int64,
        )
        return FeatureMatrix(
            values=self.values[:, columns],
            mask=self.mask[:, columns],
            block=FeatureBlock(
                names=tuple(self.block.names[index] for index in columns),
                groups=tuple(self.block.groups[index] for index in columns),
            ),
        )


@dataclass(frozen=True)
class CandidateSet:
    """The edges to be scored, their features and their ground truth."""

    #: Edge identity, matching GraphEvent.edge_key().
    keys: tuple[tuple[str, int, str], ...]
    ts: np.ndarray
    matrix: FeatureMatrix
    labels: np.ndarray
    #: Pattern name per candidate, empty string when the candidate is normal.
    patterns: tuple[str, ...]

    @property
    def n_candidates(self) -> int:
        return len(self.keys)

    def y_true(self) -> np.ndarray:
        """Ground truth as integers, the shape every metric expects."""
        return self.labels.astype(np.int8)
