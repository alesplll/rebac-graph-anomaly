"""Capability levels decide which feature groups a source can fill."""

from rga.adapters.base import Capabilities


def test_bare_snapshot_is_level_zero() -> None:
    assert Capabilities(timestamps=False, provenance=False, change_log=False).level == 0


def test_timestamps_alone_are_level_one() -> None:
    assert Capabilities(timestamps=True, provenance=False, change_log=False).level == 1


def test_timestamps_with_provenance_are_level_two() -> None:
    assert Capabilities(timestamps=True, provenance=True, change_log=True).level == 2


def test_provenance_without_timestamps_is_still_level_zero() -> None:
    # Knowing who granted a right is useless without knowing when.
    assert Capabilities(timestamps=False, provenance=True, change_log=False).level == 0


def test_a_change_log_alone_does_not_raise_the_level() -> None:
    assert Capabilities(timestamps=False, provenance=False, change_log=True).level == 0
