"""Entity id parsing. Formats are fixed by the live opens3-rebac engine."""

import pytest

from rga.domain.entities import EntityType, entity_type, parent_bucket


@pytest.mark.parametrize(
    ("entity_id", "expected"),
    [
        ("user:550e8400-e29b-41d4-a716-446655440000", EntityType.USER),
        ("group:devops", EntityType.GROUP),
        ("bucket:my-photos", EntityType.BUCKET),
        ("object:my-photos/2024/cat.jpg", EntityType.OBJECT),
    ],
)
def test_entity_type_recognises_every_prefix(entity_id: str, expected: EntityType) -> None:
    assert entity_type(entity_id) is expected


@pytest.mark.parametrize("bad", ["", "user", "user:", ":alice", "resource:r1"])
def test_entity_type_rejects_malformed_ids(bad: str) -> None:
    with pytest.raises(ValueError):
        entity_type(bad)


def test_parent_bucket_extracts_the_containing_bucket() -> None:
    assert parent_bucket("object:my-photos/2024/cat.jpg") == "bucket:my-photos"


def test_parent_bucket_handles_keys_without_slashes_in_the_name() -> None:
    assert parent_bucket("object:logs/app.log") == "bucket:logs"


def test_parent_bucket_rejects_non_objects() -> None:
    with pytest.raises(ValueError, match="not an object id"):
        parent_bucket("bucket:my-photos")
