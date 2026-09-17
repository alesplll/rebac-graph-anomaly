"""Attribution over the candidate's own features.

Gradient times input: how much the score would move per unit of a feature, times how
much of that feature this candidate actually has. It answers "which of the things we
know about this change pushed it up the list".

Unobserved features are reported as unobserved, never as a zero contribution. Zero
means the value had no influence; unobserved means the authorization engine could not
tell us, and an analyst has to be able to tell those apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from rga.features.spec import CandidateSet, FeatureGroup


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's share of the score."""

    name: str
    group: FeatureGroup
    value: float
    contribution: float
    observed: bool


def feature_contributions(
    scorer, candidates: CandidateSet, position: int, *, top: int = 10
) -> tuple[FeatureContribution, ...]:
    """The features that moved this candidate's score, strongest first.

    Empty when the scorer reads no candidate features — which is itself worth showing
    an analyst, since it says the ranking came from graph structure alone.
    """
    gradients = getattr(scorer, "feature_gradients", None)
    if gradients is None:
        return ()
    attribution = gradients(candidates, position)
    if attribution is None:
        return ()

    block = candidates.matrix.block
    values = candidates.matrix.values[position]
    observed = candidates.matrix.mask[position]

    # `dense_matrix` appends one coverage column per group after the block; those are
    # bookkeeping rather than features an analyst would recognise, so they are left out.
    found = [
        FeatureContribution(
            name=name,
            group=block.groups[column],
            value=float(values[column]),
            contribution=float(attribution[column]) if bool(observed[column]) else 0.0,
            observed=bool(observed[column]),
        )
        for column, name in enumerate(block.names)
    ]
    found.sort(key=lambda item: abs(item.contribution), reverse=True)
    return tuple(found[:top])
