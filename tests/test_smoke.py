"""The package imports and the test harness runs."""


def test_package_exposes_version() -> None:
    from rga import __version__

    assert isinstance(__version__, str)
    assert __version__.count(".") >= 1
