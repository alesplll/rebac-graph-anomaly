"""Establishing durable access.

Both patterns create new structure rather than a new edge between existing
nodes, which is what makes them a fair generalisation test: a model that learned
the first five patterns has seen nothing shaped like these.
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
    sample_night_ts,
    sample_ts,
)
from rga.util.timeutil import HOUR_MS, MINUTE_MS

#: How many buckets a shadow group takes rights on.
_SHADOW_BUCKETS = (4, 10)
#: How many people a delegation chain passes through.
_CHAIN_LENGTH = 4
#: Spacing between the steps of a chain.
_CHAIN_STEP_MS = 20 * MINUTE_MS


@register
class ShadowGroup:
    """A fresh group with a single member and administrative rights on many buckets.

    Groups exist to be shared. One that holds broad rights and has exactly one
    member is not serving the purpose groups serve.
    """

    name = "shadow_group"

    def inject(self, context: InjectionContext) -> Injection:
        users = [user for user in context.org.users if user.id in context.graph.node_index]
        if not users or len(context.org.buckets) < _SHADOW_BUCKETS[0]:
            raise NoCandidateError(self.name)

        owner = pick(context.rng, users)
        group_id = f"group:svc-{int(context.rng.integers(1 << 32)):08x}"
        window_start, window_end = context.window
        start = sample_night_ts(context.rng, (window_start, window_end - HOUR_MS))

        events = [
            GraphEvent(
                ts=start,
                op=EventOp.GRANT,
                subject=owner.id,
                relation=RelationType.MEMBER_OF,
                object=group_id,
                actor=owner.id,
            )
        ]

        count = min(
            int(context.rng.integers(_SHADOW_BUCKETS[0], _SHADOW_BUCKETS[1] + 1)),
            len(context.org.buckets),
        )
        for offset, index in enumerate(context.rng.permutation(len(context.org.buckets))[:count]):
            ts = min(
                start + (offset + 1) * int(context.rng.integers(1, 20)) * MINUTE_MS,
                window_end - 1,
            )
            events.append(
                GraphEvent(
                    ts=ts,
                    op=EventOp.GRANT,
                    subject=group_id,
                    relation=RelationType.HAS_PERMISSION,
                    object=context.org.buckets[int(index)].id,
                    level=PermissionLevel.ADMIN,
                    actor=owner.id,
                )
            )

        events.sort(key=lambda event: event.ts)
        return Injection(tuple(events), tuple(label_for(event, self.name) for event in events))


@register
class DelegationCascade:
    """Administrative rights on one bucket passed along a chain of people in an hour.

    Each step is a legitimate delegation in isolation. The chain is what is wrong:
    rights move hand to hand faster than any approval process works.
    """

    name = "delegation_cascade"

    def inject(self, context: InjectionContext) -> Injection:
        users = [user for user in context.org.users if user.id in context.graph.node_index]
        if len(users) < _CHAIN_LENGTH:
            raise NoCandidateError(self.name)

        window_start, window_end = context.window
        span = _CHAIN_LENGTH * _CHAIN_STEP_MS
        if window_end - window_start <= span:
            raise NoCandidateError(self.name)

        order = context.rng.permutation(len(users))[:_CHAIN_LENGTH]
        chain = [users[int(index)].id for index in order]
        bucket = pick(context.rng, context.org.buckets).id
        start = sample_ts(context.rng, (window_start, window_end - span))

        events = []
        actor = chain[0]
        for step in range(1, _CHAIN_LENGTH):
            events.append(
                GraphEvent(
                    ts=start + step * _CHAIN_STEP_MS,
                    op=EventOp.GRANT,
                    subject=chain[step],
                    relation=RelationType.HAS_PERMISSION,
                    object=bucket,
                    level=PermissionLevel.ADMIN,
                    actor=actor,
                )
            )
            actor = chain[step]

        return Injection(tuple(events), tuple(label_for(event, self.name) for event in events))
