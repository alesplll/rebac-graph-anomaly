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


def test_the_page_offers_a_state_filter() -> None:
    markup = (WEB / "index.html").read_text(encoding="utf-8")

    assert 'id="filter-state"' in markup


def test_each_row_gets_a_state_control() -> None:
    """A change is triaged from the list, not only from the card."""
    script = (WEB / "app.js").read_text(encoding="utf-8")

    assert "data-state" in script
    assert "/api/decisions" in script


def test_the_page_and_the_service_agree_on_the_state_names() -> None:
    """The row control must offer all three names, so it carries its own copy.

    That copy is the risk: renaming a state in the service and not in the page would
    leave the two disagreeing with nothing to notice it. This is what notices.
    """
    from rga.service.triage import TITLES

    script = (WEB / "app.js").read_text(encoding="utf-8")

    for state in ("open", "dismissed", "revoked"):
        assert TITLES[state] in script, f"{state} is called {TITLES[state]!r} by the service"


def test_the_page_reports_its_own_failures() -> None:
    """Twice a broken script left an empty screen and said nothing about why."""
    script = (WEB / "app.js").read_text(encoding="utf-8")

    assert 'addEventListener("error"' in script
    assert 'addEventListener("unhandledrejection"' in script


def test_the_choice_filters_need_no_button() -> None:
    """Picking from a list is the whole instruction; asking for a second one is noise.

    The state control already applied itself on choice, and the relation one did not,
    which read as one of them being broken.
    """
    markup = (WEB / "index.html").read_text(encoding="utf-8")
    script = (WEB / "app.js").read_text(encoding="utf-8")

    assert 'id="apply"' not in markup
    for control in ("filter-relation", "filter-state"):
        assert f'getElementById("{control}").addEventListener("change"' in script, control


def test_the_typed_search_keeps_a_button_of_its_own() -> None:
    """Typing has no end the page can see, so the search says when it is done."""
    markup = (WEB / "index.html").read_text(encoding="utf-8")
    script = (WEB / "app.js").read_text(encoding="utf-8")

    assert 'id="find"' in markup
    # The button belongs to the field: they sit in one group, in that order.
    group = markup.split('id="filter-subject"', 1)[1].split("</label>", 1)[0]
    assert 'id="find"' in group
    assert 'getElementById("find")' in script


def test_enter_searches_too() -> None:
    script = (WEB / "app.js").read_text(encoding="utf-8")

    assert "Enter" in script


def test_the_state_control_lives_on_the_card_not_in_the_list() -> None:
    """The list has one job — scanning — and little room; the card has the space."""
    script = (WEB / "app.js").read_text(encoding="utf-8")

    row = script.split("function rowMarkup", 1)[-1]
    row = script.split("body.incidents.forEach", 1)[1].split("queue.append(row)", 1)[0]
    assert "stateControl" not in row

    opened = script.split("async function openCard", 1)[1]
    assert "stateControl" in opened
