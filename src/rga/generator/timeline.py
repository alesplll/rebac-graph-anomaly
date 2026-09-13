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

#: Share of ordinary grants that land on an already-uploaded object rather than a
#: bucket. Without these, every object-level right in the journal would be issued
#: at the instant its object appeared, and the age of the target would separate
#: normal traffic from anomalies on its own — an artefact of the simulation, not a
#: property of access graphs. Found by the Module 2 baselines: an isolation forest
#: reached 0.995 ROC-AUC on it before this existed.
_OBJECT_GRANT_SHARE = 0.35

#: Share of ordinary grants a resource owner issues to themselves. Owners raising
#: their own level on something they own is routine. Without it no normal grant
#: ever has actor == subject, and that single flag would separate most anomaly
#: patterns from the rest of the journal without any model at all.
_SELF_GRANT_SHARE = 0.12


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
    uploaded: list[tuple[str, str]] = []
    # Subject, target and the moment the right was granted. The timestamp is not
    # decoration: a day's phases run in order but each event draws its own time
    # within the day, so a grant made at 21:00 can be recorded before a departure
    # drawn at 14:00 of the same day. Revoking it would put the revoke before its
    # own grant once the journal is sorted.
    direct_grants: set[tuple[str, str, int]] = set()

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
            uploaded.append((bucket_id, object_id))
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
                    # The platform writes this as part of the upload; nobody
                    # decided it, so it is attributed to the system.
                    actor=_SYSTEM,
                )
            )

        for _ in range(int(rng.poisson(config.grants_per_day * scale))):
            grant = _draw_grant(org, present, uploaded, config, rng, start, exception_rate)
            if grant is None:
                continue
            events.append(grant)
            if grant.subject.startswith("user:"):
                direct_grants.add((grant.subject, grant.object, grant.ts))

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
            # One revoke per edge, not per grant: the same right may have been
            # granted more than once, and a second revoke would hit an edge that
            # is no longer live.
            held = sorted(
                {pair[1] for pair in direct_grants if pair[0] == leaver and pair[2] < ts}
            )
            for target in held:
                events.append(
                    GraphEvent(
                        ts,
                        EventOp.REVOKE,
                        leaver,
                        RelationType.HAS_PERMISSION,
                        target,
                        actor=_SYSTEM,
                    )
                )
            direct_grants = {
                pair for pair in direct_grants if not (pair[0] == leaver and pair[1] in held)
            }
            present.discard(leaver)


def _draw_grant(
    org: Organization,
    present: set[str],
    uploaded: list[tuple[str, str]],
    config: TimelineConfig,
    rng: np.random.Generator,
    day_start_ts: int,
    exception_rate: float,
) -> GraphEvent | None:
    """One right granted by the normal procedure, or a legitimate exception.

    The procedure grants to the team group. The exception grants directly to a
    person, sometimes across departments — rare, but normal. Either can land on a
    bucket or on an object that was uploaded some time ago.
    """
    if not present:
        return None
    ts = sample_time_of_day(rng, day_start_ts, config)

    if uploaded and rng.random() < _OBJECT_GRANT_SHARE:
        bucket_id, target = uploaded[int(rng.integers(len(uploaded)))]
        bucket = org.bucket(bucket_id)
    else:
        bucket = org.buckets[int(rng.integers(len(org.buckets)))]
        target = bucket.id
    approver = bucket.owner

    if rng.random() < _SELF_GRANT_SHARE and bucket.owner in present:
        return GraphEvent(
            ts,
            EventOp.GRANT,
            bucket.owner,
            RelationType.HAS_PERMISSION,
            target,
            _TEAM_LEVELS[int(rng.integers(len(_TEAM_LEVELS)))],
            actor=bucket.owner,
        )

    if rng.random() >= exception_rate:
        return GraphEvent(
            ts,
            EventOp.GRANT,
            org.team(bucket.team).group_id,
            RelationType.HAS_PERMISSION,
            target,
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
        target,
        PermissionLevel.READ,
        actor=approver,
    )
