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


def test_the_page_carries_the_console_furniture() -> None:
    """A monitoring tool needs a filter bar, tabs and a selection bar."""
    markup = (WEB / "index.html").read_text(encoding="utf-8")

    for marker in ('id="filters"', 'id="queue"', 'id="selection"', 'id="history"', 'data-tab='):
        assert marker in markup, marker


def test_there_is_exactly_one_theme() -> None:
    """A switch is one more thing to be in the wrong state when the projector is on.

    The palette lives in a single token block; nothing chooses between two of them.
    """
    css = (WEB / "style.css").read_text(encoding="utf-8")
    script = (WEB / "app.js").read_text(encoding="utf-8")

    assert ":root" in css
    assert "data-theme" not in css
    assert "data-theme" not in script


def test_the_stylesheet_paints_through_tokens() -> None:
    """A colour written into a rule would survive the theme switch and look wrong.

    Only the two `:root` blocks may name a colour; everything else refers to them.
    Identifier selectors such as `#queue` are not colours and must not trip this.
    """
    import re

    css = (WEB / "style.css").read_text(encoding="utf-8")
    without_tokens = re.sub(r":root[^{]*\{[^}]*\}", "", css)

    leftover = re.findall(r"#[0-9a-fA-F]{3,8}(?![0-9A-Za-z_-])", without_tokens)

    assert not leftover, f"literal colours outside the token blocks: {leftover}"


def test_the_token_check_would_notice() -> None:
    """The regex must not be so narrow that it matches nothing."""
    import re

    assert re.findall(r"#[0-9a-fA-F]{3,8}(?![0-9A-Za-z_-])", "a { color: #b3261e; }")
    assert not re.findall(r"#[0-9a-fA-F]{3,8}(?![0-9A-Za-z_-])", "#queue { margin: 0; }")


def test_the_script_wires_the_console_together() -> None:
    """The page cannot be driven by a browser here, so the wiring is read as text."""
    script = (WEB / "app.js").read_text(encoding="utf-8")

    for marker in (
        "/api/decisions",
        "state.selected",
        "data-outcome",
        "data-pick",
        "renderHistory",
        "shiftKey",
    ):
        assert marker in script, marker


def test_every_identifier_reaching_the_markup_is_escaped() -> None:
    """Node names come from somebody else's engine and land inside HTML."""
    script = (WEB / "app.js").read_text(encoding="utf-8")

    assert "function escapeHtml" in script
    # The queue row, the history row and the card all build markup from identifiers.
    for builder in ("function rowMarkup", "function decisionRow", "function node"):
        body = script.split(builder, 1)[1].split("\n}", 1)[0]
        assert "escapeHtml" in body, builder
