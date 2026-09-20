"""The page is served, self-contained, and says nothing it should not."""

from pathlib import Path

import pytest

from rga.explain.text import FORBIDDEN_WORDS

WEB = Path("web")


def test_the_page_exists() -> None:
    assert (WEB / "index.html").is_file()
    assert (WEB / "style.css").is_file()
    assert (WEB / "app.js").is_file()


def test_nothing_is_loaded_from_the_network() -> None:
    """A defence room may have no internet, and a CDN is a point of failure."""
    markup = (WEB / "index.html").read_text(encoding="utf-8")

    assert "http://" not in markup
    assert "https://" not in markup


@pytest.mark.parametrize("name", ["index.html", "app.js"])
def test_the_page_pronounces_no_verdict(name) -> None:
    text = (WEB / name).read_text(encoding="utf-8").lower()

    for word in FORBIDDEN_WORDS:
        assert word not in text
