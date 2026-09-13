"""Vertical escalation.

Both patterns model a subject acquiring rights through a path the standard
procedure does not provide. What makes them detectable structurally is not the
level itself — plenty of people legitimately hold admin — but that the right
appears without the surrounding structure that normally accompanies it.
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
    level_on,
    pick,
    register,
    sample_maybe_night_ts,
)


@register
class SelfGrantAdmin:
    """A user grants itself administrative rights on a bucket of its own team.

    The subject already has ordinary access through the team group, so the new
    edge is not an impossible one — it is one that nobody else in a comparable
    position holds, and that the subject issued for itself.
    """

    name = "self_grant_admin"

    def inject(self, context: InjectionContext) -> Injection:
        users = [user for user in context.org.users if user.id in context.graph.node_index]
        if not users:
            raise NoCandidateError(self.name)

        for _ in range(ATTEMPTS):
            user = pick(context.rng, users)
            team = context.org.team_of(user.id)
            if not team.buckets:
                continue
            bucket = pick(context.rng, team.buckets)
            if level_on(context.graph, user.id, bucket) >= PermissionLevel.ADMIN:
                continue

            event = GraphEvent(
                ts=sample_maybe_night_ts(context.rng, context.window),
                op=EventOp.GRANT,
                subject=user.id,
                relation=RelationType.HAS_PERMISSION,
                object=bucket,
                level=PermissionLevel.ADMIN,
                actor=user.id,
            )
            return Injection((event,), (label_for(event, self.name),))

        raise NoCandidateError(self.name)


@register
class PrivilegedGroupJoin:
    """A user joins a cross-cutting group holding elevated rights.

    The group is legitimate and so is membership in it — for the people who went
    through the process. The anomaly is the membership appearing on its own,
    issued by the joiner, without the department or role that normally precedes it.
    """

    name = "privileged_group_join"

    def inject(self, context: InjectionContext) -> Injection:
        if not context.org.cross_cutting:
            raise NoCandidateError(self.name)
        users = [user for user in context.org.users if user.id in context.graph.node_index]
        if not users:
            raise NoCandidateError(self.name)

        for _ in range(ATTEMPTS):
            group = pick(context.rng, context.org.cross_cutting)
            user = pick(context.rng, users)
            if user.id in group.members:
                continue

            event = GraphEvent(
                ts=sample_maybe_night_ts(context.rng, context.window),
                op=EventOp.GRANT,
                subject=user.id,
                relation=RelationType.MEMBER_OF,
                object=group.id,
                actor=user.id,
            )
            return Injection((event,), (label_for(event, self.name),))

        raise NoCandidateError(self.name)
