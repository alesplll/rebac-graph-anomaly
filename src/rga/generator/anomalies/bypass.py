"""Bypassing the delegation hierarchy.

The actor in both patterns is a plausible approver rather than the subject. If
every pattern made the subject its own actor, provenance alone would separate
anomalies from normal data and the structural model would never be exercised.
"""

from __future__ import annotations

from rga.domain.entities import EntityType, entity_type, parent_bucket
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
    sample_ts,
)


@register
class HierarchyBypass:
    """A right on an object, granted to someone with no right on its bucket.

    Object-level rights are ordinary here — the engine does not inherit through
    containment, so every upload produces one. What is not ordinary is who holds
    it: normally the right goes to the group that owns the containing bucket, and
    whoever holds it on the object holds something on the bucket too. This edge
    lands on a leaf, held by a subject with nothing above it.
    """

    name = "hierarchy_bypass"

    def inject(self, context: InjectionContext) -> Injection:
        objects = [
            node_id
            for node_id in context.graph.node_ids
            if entity_type(node_id) is EntityType.OBJECT
        ]
        if not objects:
            raise NoCandidateError(self.name)
        users = [user for user in context.org.users if user.id in context.graph.node_index]

        for _ in range(ATTEMPTS):
            target = pick(context.rng, objects)
            bucket_id = parent_bucket(target)
            user = pick(context.rng, users)
            if level_on(context.graph, user.id, bucket_id) is not PermissionLevel.NONE:
                continue
            if level_on(context.graph, user.id, target) is not PermissionLevel.NONE:
                continue
            try:
                approver = context.org.bucket(bucket_id).owner
            except KeyError:
                continue
            if approver == user.id:
                continue

            event = GraphEvent(
                ts=sample_ts(context.rng, context.window),
                op=EventOp.GRANT,
                subject=user.id,
                relation=RelationType.HAS_PERMISSION,
                object=target,
                level=PermissionLevel.WRITE,
                actor=approver,
            )
            return Injection((event,), (label_for(event, self.name),))

        raise NoCandidateError(self.name)


@register
class CrossDepartment:
    """A right on another department's bucket, with no shared structure to justify it.

    Rights across departments exist in the normal data too — that is what the
    legitimate-exception rate produces. The difference is the surrounding
    structure: a normal exception sits near shared groups and shared projects,
    this one sits alone.
    """

    name = "cross_department"

    def inject(self, context: InjectionContext) -> Injection:
        users = [user for user in context.org.users if user.id in context.graph.node_index]

        for _ in range(ATTEMPTS):
            user = pick(context.rng, users)
            department = context.org.department_of(user.id)
            foreign = [
                bucket
                for bucket in context.org.buckets
                if context.org.team(bucket.team).department != department
            ]
            if not foreign:
                continue
            bucket = pick(context.rng, foreign)
            if level_on(context.graph, user.id, bucket.id) is not PermissionLevel.NONE:
                continue

            target_department = context.org.team(bucket.team).department
            approvers = [
                candidate.id
                for candidate in context.org.users
                if candidate.department == target_department
                and candidate.id != bucket.owner
                and candidate.id in context.graph.node_index
            ]
            if not approvers:
                continue

            event = GraphEvent(
                ts=sample_ts(context.rng, context.window),
                op=EventOp.GRANT,
                subject=user.id,
                relation=RelationType.HAS_PERMISSION,
                object=bucket.id,
                level=PermissionLevel.WRITE,
                actor=pick(context.rng, approvers),
            )
            return Injection((event,), (label_for(event, self.name),))

        raise NoCandidateError(self.name)
