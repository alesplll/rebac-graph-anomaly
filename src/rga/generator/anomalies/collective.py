"""Patterns that exist only in a group of edges.

Module 3 measured a generalisation gap of +0.02 between the five patterns the
supervised model trains on and the three hidden from it, and concluded that the
check proves almost nothing: the hidden patterns are shaped like the trained ones,
so a template learned on one transfers to the other. These two are the answer to
that. Every edge they create is, on its own, an ordinary cross-team grant at an
ordinary level during working hours. What is wrong lives one level up, in how the
edges sit together — a closed cycle in the first case, a convergence on a single
object in the second.

They are used only by `configs/generator/small-divergent.yaml`. The datasets the
published tables were measured on keep the original eight patterns untouched.
"""

from __future__ import annotations

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.anomalies.base import (
    Injection,
    InjectionContext,
    NoCandidateError,
    label_for,
    pick,
    register,
    sample_business_ts,
)
from rga.util.timeutil import DAY_MS

#: How many people a ring passes through.
_RING_SIZE = (3, 4)
#: How many people converge on one object.
_CONVERGING = (5, 9)
#: Spacing between the steps of a ring, so it is not a burst either.
_RING_STEP_MS = 7 * 60 * 60 * 1000
#: Span the convergence is spread over, in days.
_CONVERGENCE_DAYS = 4


def _users_in_graph(context: InjectionContext) -> list[str]:
    return [user.id for user in context.org.users if user.id in context.graph.node_index]


@register
class MutualGrantRing:
    """A closed cycle of grants: each person opens their team's data to the next.

    Every other pattern in the catalogue is a star, a chain or a single edge, and
    every one of them has a centre — one subject, one object, or one edge. A cycle
    has none. Each edge of it is an ordinary cross-team grant at write level during
    working hours, and a detector reading one change at a time has nothing to see;
    only the closure, where the last grant points back at the first giver, makes the
    group of them strange.
    """

    name = "mutual_grant_ring"

    def inject(self, context: InjectionContext) -> Injection:
        size = int(context.rng.integers(_RING_SIZE[0], _RING_SIZE[1] + 1))
        window_start, window_end = context.window
        span = size * _RING_STEP_MS
        if window_end - window_start <= span:
            raise NoCandidateError(self.name)

        # One member per team, so that every grant of the ring crosses a boundary and
        # no two members can offer the same bucket.
        by_team: dict[str, list[str]] = {}
        for user in _users_in_graph(context):
            by_team.setdefault(context.org.team_of(user).id, []).append(user)
        teams = [team for team in by_team if context.org.team(team).buckets]
        if len(teams) < size:
            raise NoCandidateError(self.name)

        order = context.rng.permutation(len(teams))[:size]
        chosen = [teams[int(index)] for index in order]
        ring = [pick(context.rng, by_team[team]) for team in chosen]
        # The bucket each person opens is one of their own team's, so the grant they
        # make is exactly the cross-team grant an ordinary colleague would make.
        offered = [pick(context.rng, context.org.team(team).buckets) for team in chosen]

        start = sample_business_ts(context.rng, (window_start, window_end - span))
        events = [
            GraphEvent(
                ts=start + step * _RING_STEP_MS,
                op=EventOp.GRANT,
                subject=ring[(step + 1) % size],
                relation=RelationType.HAS_PERMISSION,
                object=offered[step],
                level=PermissionLevel.WRITE,
                actor=ring[step],
            )
            for step in range(size)
        ]

        events.sort(key=lambda event: event.ts)
        return Injection(tuple(events), tuple(label_for(event, self.name) for event in events))


@register
class ConvergentAccess:
    """Many unrelated people quietly granted read access on one bucket over days.

    Deliberately the transpose of `grant_burst`, which the supervised model does
    train on: that one is a star out of a single subject compressed into minutes,
    this one is a star into a single object spread over days. The two share a shape
    only if direction and timing are ignored, which makes this the sharper question
    to ask a model that claims to have learned structure.
    """

    name = "convergent_access"

    def inject(self, context: InjectionContext) -> Injection:
        users = _users_in_graph(context)
        count = int(context.rng.integers(_CONVERGING[0], _CONVERGING[1] + 1))
        if len(users) < count or not context.org.buckets:
            raise NoCandidateError(self.name)

        window_start, window_end = context.window
        span = _CONVERGENCE_DAYS * DAY_MS
        if window_end - window_start <= span:
            raise NoCandidateError(self.name)

        target = pick(context.rng, context.org.buckets)
        owning_team = target.team
        outsiders = [
            user for user in users if context.org.team_of(user).id != owning_team
        ]
        if len(outsiders) < count:
            raise NoCandidateError(self.name)

        order = context.rng.permutation(len(outsiders))[:count]
        newcomers = [outsiders[int(index)] for index in order]
        approver = pick(context.rng, [user for user in users if user not in newcomers])

        start = sample_business_ts(context.rng, (window_start, window_end - span))
        step = span // count
        events = [
            GraphEvent(
                ts=min(start + position * step, window_end - 1),
                op=EventOp.GRANT,
                subject=newcomer,
                relation=RelationType.HAS_PERMISSION,
                object=target.id,
                level=PermissionLevel.READ,
                actor=approver,
            )
            for position, newcomer in enumerate(newcomers)
        ]

        events.sort(key=lambda event: event.ts)
        return Injection(tuple(events), tuple(label_for(event, self.name) for event in events))
