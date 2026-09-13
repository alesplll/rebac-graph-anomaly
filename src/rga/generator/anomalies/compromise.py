"""Account compromise.

Neither pattern creates an impossible right. What marks them is the rhythm: a
volume of grants no single person produces in an hour, or activity from an
account that has been silent for weeks.
"""

from __future__ import annotations

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.anomalies.base import (
    ATTEMPTS,
    Injection,
    InjectionContext,
    NoCandidateError,
    label_for,
    last_activity,
    level_on,
    pick,
    register,
    sample_maybe_night_ts,
)
from rga.util.timeutil import DAY_MS, MINUTE_MS

#: How many grants a compromised account issues.
_BURST_SIZE = (6, 16)
#: How long the burst lasts.
_BURST_SPAN_MS = 45 * MINUTE_MS
#: Silence after which an account counts as dormant.
_DORMANT_MS = 21 * DAY_MS


@register
class GrantBurst:
    """One subject grants itself rights on many foreign buckets within an hour."""

    name = "grant_burst"

    def inject(self, context: InjectionContext) -> Injection:
        users = [user for user in context.org.users if user.id in context.graph.node_index]
        window_start, window_end = context.window
        if window_end - window_start <= _BURST_SPAN_MS:
            raise NoCandidateError(self.name)

        for _ in range(ATTEMPTS):
            user = pick(context.rng, users)
            own_team = context.org.team_of(user.id).id
            foreign = [bucket.id for bucket in context.org.buckets if bucket.team != own_team]
            if len(foreign) < _BURST_SIZE[0]:
                continue

            count = min(
                int(context.rng.integers(_BURST_SIZE[0], _BURST_SIZE[1] + 1)), len(foreign)
            )
            order = context.rng.permutation(len(foreign))[:count]
            start = sample_maybe_night_ts(context.rng, (window_start, window_end - _BURST_SPAN_MS))

            events = [
                GraphEvent(
                    ts=start + int(context.rng.integers(_BURST_SPAN_MS)),
                    op=EventOp.GRANT,
                    subject=user.id,
                    relation=RelationType.HAS_PERMISSION,
                    object=foreign[int(index)],
                    level=PermissionLevel.WRITE,
                    actor=user.id,
                )
                for index in order
            ]
            events.sort(key=lambda event: event.ts)
            return Injection(
                tuple(events), tuple(label_for(event, self.name) for event in events)
            )

        raise NoCandidateError(self.name)


@register
class DormantAwakening:
    """An account silent for weeks suddenly takes an elevated right."""

    name = "dormant_awakening"

    def inject(self, context: InjectionContext) -> Injection:
        threshold = context.window[0] - _DORMANT_MS
        dormant = [
            user
            for user in context.org.users
            if user.id in context.graph.node_index
            and 0 <= last_activity(context.graph, user.id) <= threshold
        ]
        if not dormant:
            raise NoCandidateError(self.name)

        for _ in range(ATTEMPTS):
            user = pick(context.rng, dormant)
            bucket = pick(context.rng, context.org.buckets)
            if level_on(context.graph, user.id, bucket.id) >= PermissionLevel.WRITE:
                continue

            event = GraphEvent(
                ts=sample_maybe_night_ts(context.rng, context.window),
                op=EventOp.GRANT,
                subject=user.id,
                relation=RelationType.HAS_PERMISSION,
                object=bucket.id,
                level=PermissionLevel.WRITE,
                actor=user.id,
            )
            return Injection((event,), (label_for(event, self.name),))

        raise NoCandidateError(self.name)
