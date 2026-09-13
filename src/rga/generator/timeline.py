"""The growth process: an organization living through time.

Day zero founds the company — a share of the staff, the team groups, the first
projects. After that each day draws hires, departures, uploads and grants from
Poisson counts, and the normal procedure for granting a right is always the same:
rights go to the team group, not to individuals. Deviations from that procedure
exist in the normal data too, at the configured rate, because an organization
where the procedure is never bypassed is not a realistic baseline.

Objects receive their own permission edges. The target engine does not inherit
rights through containment — a permission on a bucket does not cover its objects,
which is a deliberate decision of that project, verified against its test suite —
so the gateway writes an explicit permission after every upload. Reproducing that
matters: if the normal graph had no object-level rights, the bypass pattern would
be detectable by the mere presence of one.
"""

from __future__ import annotations

import numpy as np

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.config import TimelineConfig
from rga.generator.org import Organization
from rga.util.timeutil import day_start, is_weekend, sample_time_of_day

#: Share of the staff already present when the timeline starts.
_FOUNDING_SHARE = 0.6

#: Levels a team group normally holds on its own resources.
_TEAM_LEVELS = (PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.CREATE)

#: Identity used for changes the platform itself performs.
_SYSTEM = "user:system"


def generate_normal_journal(
    org: Organization,
    config: TimelineConfig,
    rng: np.random.Generator,
    *,
    exception_rate: float,
) -> list[GraphEvent]:
    """Produce the journal of a normally operating organization.

    `exception_rate` is the share of grants made outside the standard group
    procedure. It describes the company's habits rather than the simulation's
    intensity, so it lives in OrgConfig and is passed in explicitly.
    """
    events: list[GraphEvent] = []
    present: set[str] = set()
    pending_users = list(org.users)
    rng.shuffle(pending_users)  # type: ignore[arg-type]

    founding_count = max(1, int(len(pending_users) * _FOUNDING_SHARE))
    founders, newcomers = pending_users[:founding_count], pending_users[founding_count:]

    _found(events, org, founders, present, config, rng)
    _operate(events, org, newcomers, present, config, rng, exception_rate)

    events.sort(key=lambda event: event.ts)
    return events


def _found(
    events: list[GraphEvent],
    org: Organization,
    founders: list,
    present: set[str],
    config: TimelineConfig,
    rng: np.random.Generator,
) -> None:
    """Day zero: staff, groups, buckets and the standard grants."""
    start = day_start(config.start_ts, 0)

    for user in founders:
        events.append(
            GraphEvent(
                sample_time_of_day(rng, start, config),
                EventOp.GRANT,
                user.id,
                RelationType.MEMBER_OF,
                org.team(user.team).group_id,
                actor=_SYSTEM,
            )
        )
        present.add(user.id)

    for bucket in org.buckets:
        ts = sample_time_of_day(rng, start, config)
        events.append(
            GraphEvent(
                ts, EventOp.GRANT, bucket.owner, RelationType.OWNER_OF, bucket.id, actor=_SYSTEM
            )
        )
        events.append(
            GraphEvent(
                ts,
                EventOp.GRANT,
                org.team(bucket.team).group_id,
                RelationType.HAS_PERMISSION,
                bucket.id,
                _TEAM_LEVELS[int(rng.integers(len(_TEAM_LEVELS)))],
                actor=bucket.owner,
            )
        )

    for group in org.cross_cutting:
        for member in group.members:
            events.append(
                GraphEvent(
                    sample_time_of_day(rng, start, config),
                    EventOp.GRANT,
                    member,
                    RelationType.MEMBER_OF,
                    group.id,
                    actor=_SYSTEM,
                )
            )
        for bucket_id in group.buckets:
            events.append(
                GraphEvent(
                    sample_time_of_day(rng, start, config),
                    EventOp.GRANT,
                    group.id,
                    RelationType.HAS_PERMISSION,
                    bucket_id,
                    group.level,
                    actor=_SYSTEM,
                )
            )


def _operate(
    events: list[GraphEvent],
    org: Organization,
    newcomers: list,
    present: set[str],
    config: TimelineConfig,
    rng: np.random.Generator,
    exception_rate: float,
) -> None:
    """Days one onwards: hires, uploads, grants, departures."""
    queue = list(newcomers)
    pending_objects = [(bucket.id, obj) for bucket in org.buckets for obj in bucket.objects]
    rng.shuffle(pending_objects)  # type: ignore[arg-type]
    direct_grants: set[tuple[str, str]] = set()

    for day in range(1, config.days):
        start = day_start(config.start_ts, day)
        scale = config.weekend_rate if is_weekend(start) else 1.0

        for _ in range(int(rng.poisson(config.hires_per_day * scale))):
            if not queue:
                break
            user = queue.pop()
            events.append(
                GraphEvent(
                    sample_time_of_day(rng, start, config),
                    EventOp.GRANT,
                    user.id,
                    RelationType.MEMBER_OF,
                    org.team(user.team).group_id,
                    actor=_SYSTEM,
                )
            )
            present.add(user.id)

        for _ in range(int(rng.poisson(config.uploads_per_day * scale))):
            if not pending_objects:
                break
            bucket_id, object_id = pending_objects.pop()
            ts = sample_time_of_day(rng, start, config)
            events.append(
                GraphEvent(
                    ts, EventOp.GRANT, bucket_id, RelationType.PARENT_OF, object_id, actor=_SYSTEM
                )
            )
            bucket = org.bucket(bucket_id)
            events.append(
                GraphEvent(
                    ts,
                    EventOp.GRANT,
                    org.team(bucket.team).group_id,
                    RelationType.HAS_PERMISSION,
                    object_id,
                    _TEAM_LEVELS[int(rng.integers(len(_TEAM_LEVELS)))],
                    actor=bucket.owner,
                )
            )

        for _ in range(int(rng.poisson(config.grants_per_day * scale))):
            grant = _draw_grant(org, present, config, rng, start, exception_rate)
            if grant is None:
                continue
            events.append(grant)
            if grant.subject.startswith("user:"):
                direct_grants.add((grant.subject, grant.object))

        for _ in range(int(rng.poisson(config.departures_per_day * scale))):
            if not present:
                break
            leaver = sorted(present)[int(rng.integers(len(present)))]
            ts = sample_time_of_day(rng, start, config)
            events.append(
                GraphEvent(
                    ts,
                    EventOp.REVOKE,
                    leaver,
                    RelationType.MEMBER_OF,
                    org.team_of(leaver).group_id,
                    actor=_SYSTEM,
                )
            )
            for subject, target in sorted(pair for pair in direct_grants if pair[0] == leaver):
                events.append(
                    GraphEvent(
                        ts,
                        EventOp.REVOKE,
                        subject,
                        RelationType.HAS_PERMISSION,
                        target,
                        actor=_SYSTEM,
                    )
                )
                direct_grants.discard((subject, target))
            present.discard(leaver)


def _draw_grant(
    org: Organization,
    present: set[str],
    config: TimelineConfig,
    rng: np.random.Generator,
    day_start_ts: int,
    exception_rate: float,
) -> GraphEvent | None:
    """One right granted by the normal procedure, or a legitimate exception.

    The procedure grants to the team group. The exception grants directly to a
    person, sometimes across departments — rare, but normal.
    """
    if not present:
        return None
    ts = sample_time_of_day(rng, day_start_ts, config)
    bucket = org.buckets[int(rng.integers(len(org.buckets)))]
    approver = bucket.owner

    if rng.random() >= exception_rate:
        return GraphEvent(
            ts,
            EventOp.GRANT,
            org.team(bucket.team).group_id,
            RelationType.HAS_PERMISSION,
            bucket.id,
            _TEAM_LEVELS[int(rng.integers(len(_TEAM_LEVELS)))],
            actor=approver,
        )

    candidates = sorted(present)
    subject = candidates[int(rng.integers(len(candidates)))]
    return GraphEvent(
        ts,
        EventOp.GRANT,
        subject,
        RelationType.HAS_PERMISSION,
        bucket.id,
        PermissionLevel.READ,
        actor=approver,
    )
