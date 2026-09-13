"""The static structure of the modelled organization.

Departments hold teams, teams own projects, a project is a bucket holding
objects. A team maps to a group, and rights are normally granted to that group
rather than to individuals. Cross-cutting groups — an operations team, say —
hold elevated rights across many buckets, and a small share of users hold rights
outside their own department. Both exist so that the normal graph is not a
perfect tree: without them every deviation from the hierarchy would be anomalous
by construction and the problem would be trivial.

This module describes only what exists. When each fact becomes true is decided
by the timeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from rga.domain.relations import PermissionLevel
from rga.generator.config import OrgConfig

#: Probability that a cross-cutting group covers any given bucket.
_CROSS_CUTTING_COVERAGE = 0.5


@dataclass(frozen=True)
class User:
    id: str
    team: str
    department: int


@dataclass(frozen=True)
class Team:
    id: str
    group_id: str
    department: int
    members: tuple[str, ...]
    buckets: tuple[str, ...]


@dataclass(frozen=True)
class Bucket:
    id: str
    owner: str
    team: str
    objects: tuple[str, ...]


@dataclass(frozen=True)
class CrossCuttingGroup:
    id: str
    members: tuple[str, ...]
    level: PermissionLevel
    buckets: tuple[str, ...]


@dataclass(frozen=True, eq=True)
class Organization:
    """Everything that exists, without any notion of when it appeared."""

    users: tuple[User, ...]
    teams: tuple[Team, ...]
    buckets: tuple[Bucket, ...]
    cross_cutting: tuple[CrossCuttingGroup, ...]

    def user(self, entity_id: str) -> User:
        return self._users_by_id[entity_id]

    def team(self, team_id: str) -> Team:
        return self._teams_by_id[team_id]

    def bucket(self, bucket_id: str) -> Bucket:
        return self._buckets_by_id[bucket_id]

    def team_of(self, user_id: str) -> Team:
        return self.team(self.user(user_id).team)

    def department_of(self, user_id: str) -> int:
        return self.user(user_id).department

    @cached_property
    def _users_by_id(self) -> dict[str, User]:
        return {user.id: user for user in self.users}

    @cached_property
    def _teams_by_id(self) -> dict[str, Team]:
        return {team.id: team for team in self.teams}

    @cached_property
    def _buckets_by_id(self) -> dict[str, Bucket]:
        return {bucket.id: bucket for bucket in self.buckets}


def _uuid(rng: np.random.Generator) -> str:
    """A deterministic UUID drawn from the seeded generator."""
    return str(uuid.UUID(bytes=bytes(rng.integers(0, 256, size=16, dtype=np.uint8)), version=4))


def _between(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    """Inclusive integer draw."""
    low, high = bounds
    return int(rng.integers(low, high + 1))


def build_organization(config: OrgConfig, rng: np.random.Generator) -> Organization:
    """Construct the organization deterministically from a seeded generator."""
    users: list[User] = []
    teams: list[Team] = []
    buckets: list[Bucket] = []

    for department in range(config.departments):
        for team_number in range(_between(rng, config.teams_per_department)):
            team_id = f"team-{department}-{team_number}"

            members = tuple(
                f"user:{_uuid(rng)}" for _ in range(_between(rng, config.users_per_team))
            )
            users.extend(
                User(id=member, team=team_id, department=department) for member in members
            )

            team_buckets: list[str] = []
            for project in range(_between(rng, config.projects_per_team)):
                bucket_name = f"{team_id}-p{project}"
                objects = tuple(
                    f"object:{bucket_name}/file-{index:05d}.dat"
                    for index in range(_between(rng, config.objects_per_bucket))
                )
                owner = members[int(rng.integers(len(members)))]
                buckets.append(
                    Bucket(id=f"bucket:{bucket_name}", owner=owner, team=team_id, objects=objects)
                )
                team_buckets.append(f"bucket:{bucket_name}")

            teams.append(
                Team(
                    id=team_id,
                    group_id=f"group:{team_id}",
                    department=department,
                    members=members,
                    buckets=tuple(team_buckets),
                )
            )

    cross_cutting: list[CrossCuttingGroup] = []
    all_bucket_ids = tuple(bucket.id for bucket in buckets)
    for index in range(config.cross_cutting_groups):
        membership = tuple(
            user.id for user in users if rng.random() < config.cross_cutting_membership_rate
        )
        covered = tuple(
            bucket_id for bucket_id in all_bucket_ids if rng.random() < _CROSS_CUTTING_COVERAGE
        )
        cross_cutting.append(
            CrossCuttingGroup(
                id=f"group:ops-{index}",
                members=membership,
                level=PermissionLevel.WRITE if index % 2 else PermissionLevel.ADMIN,
                buckets=covered,
            )
        )

    return Organization(
        users=tuple(users),
        teams=tuple(teams),
        buckets=tuple(buckets),
        cross_cutting=tuple(cross_cutting),
    )
