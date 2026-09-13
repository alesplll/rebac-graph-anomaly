"""Entity identifiers.

The textual format is dictated by the live opens3-rebac engine and is reproduced
verbatim so that a graph read from it needs no translation:

    user:<uuid>              group:<name>
    bucket:<name>            object:<bucket>/<key>
"""

from __future__ import annotations

from enum import IntEnum


class EntityType(IntEnum):
    """Node kinds. Values are stable and index model parameters."""

    USER = 1
    GROUP = 2
    BUCKET = 3
    OBJECT = 4


_PREFIX_TO_TYPE = {
    "user": EntityType.USER,
    "group": EntityType.GROUP,
    "bucket": EntityType.BUCKET,
    "object": EntityType.OBJECT,
}


def entity_type(entity_id: str) -> EntityType:
    """Infer the node kind from an entity id prefix."""
    prefix, separator, rest = entity_id.partition(":")
    if not separator or not rest:
        raise ValueError(f"malformed entity id: {entity_id!r}")
    try:
        return _PREFIX_TO_TYPE[prefix]
    except KeyError:
        raise ValueError(f"unknown entity prefix: {prefix!r}") from None


def parent_bucket(object_id: str) -> str:
    """Return the bucket containing an object.

    The key may itself contain slashes; only the first segment is the bucket.
    """
    if entity_type(object_id) is not EntityType.OBJECT:
        raise ValueError(f"not an object id: {object_id!r}")
    path = object_id.split(":", 1)[1]
    bucket, separator, _ = path.partition("/")
    if not separator or not bucket:
        raise ValueError(f"object id has no bucket separator: {object_id!r}")
    return f"bucket:{bucket}"
