"""Static organization structure."""

import numpy as np

from rga.domain.entities import EntityType, entity_type
from rga.generator.config import OrgConfig
from rga.generator.org import build_organization

CONFIG = OrgConfig(
    departments=3,
    teams_per_department=(2, 4),
    users_per_team=(3, 6),
    projects_per_team=(1, 3),
    objects_per_bucket=(5, 20),
    cross_cutting_groups=2,
    cross_cutting_membership_rate=0.1,
    legitimate_exception_rate=0.05,
)


def _org(seed: int = 7):
    return build_organization(CONFIG, np.random.default_rng(seed))


def test_team_count_is_within_the_configured_range() -> None:
    org = _org()
    per_department: dict[int, int] = {}
    for team in org.teams:
        per_department[team.department] = per_department.get(team.department, 0) + 1
    assert len(per_department) == CONFIG.departments
    assert all(2 <= count <= 4 for count in per_department.values())


def test_every_user_belongs_to_exactly_one_team() -> None:
    org = _org()
    assert len({user.id for user in org.users}) == len(org.users)
    for user in org.users:
        owning = [team for team in org.teams if user.id in team.members]
        assert len(owning) == 1
        assert owning[0].id == user.team


def test_ids_use_the_engine_format() -> None:
    org = _org()
    assert all(entity_type(user.id) is EntityType.USER for user in org.users)
    assert all(entity_type(team.group_id) is EntityType.GROUP for team in org.teams)
    assert all(entity_type(bucket.id) is EntityType.BUCKET for bucket in org.buckets)
    assert all(
        entity_type(obj) is EntityType.OBJECT for bucket in org.buckets for obj in bucket.objects
    )


def test_objects_live_in_their_own_bucket() -> None:
    org = _org()
    for bucket in org.buckets:
        prefix = f"object:{bucket.id.split(':', 1)[1]}/"
        assert all(obj.startswith(prefix) for obj in bucket.objects)


def test_each_bucket_is_owned_by_a_member_of_its_team() -> None:
    org = _org()
    for bucket in org.buckets:
        assert bucket.owner in org.team(bucket.team).members


def test_generation_is_reproducible() -> None:
    assert _org(7) == _org(7)


def test_different_seeds_give_different_organizations() -> None:
    assert _org(7) != _org(8)


def test_lookups_resolve() -> None:
    org = _org()
    user = org.users[0]
    assert org.user(user.id) is user
    assert org.team_of(user.id).id == user.team
    assert org.department_of(user.id) == user.department
