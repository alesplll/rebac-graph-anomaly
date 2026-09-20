"""The neighbourhood drawing, rendered where it can be tested."""

import re
from pathlib import Path

import pytest

from rga.explain.incident import build_incident
from rga.explain.picture import TYPE_COLOUR, render_subgraph
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.supervised import SupervisedGnnScorer

CONFIG = load_dataset_config(Path("configs/generator/small-history.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3)


@pytest.fixture(scope="module")
def card():
    dataset = build_dataset(CONFIG)
    scorer = SupervisedGnnScorer(seed=0, config=FAST)
    scorer.fit(build_candidates(dataset, Span.TRAIN))
    evaluation = build_candidates(dataset, Span.EVAL)
    return build_incident(scorer, evaluation, 0, score=0.9, rank=1, cap=40).as_dict()


def test_it_renders_one_well_formed_svg(card) -> None:
    markup = render_subgraph(card)

    assert markup.startswith("<svg")
    assert markup.rstrip().endswith("</svg>")
    assert markup.count("<svg") == 1
    assert "viewBox" in markup


def test_it_stays_small_enough_to_read(card) -> None:
    markup = render_subgraph(card, edges=10)

    # One rect per node, and a readable picture means a dozen of them, not fifty.
    assert 2 <= markup.count("<rect") <= 16


def test_the_scored_change_is_drawn_apart_from_the_rest(card) -> None:
    markup = render_subgraph(card)

    assert 'class="change"' in markup
    assert card["subject"] in markup
    assert card["object"] in markup


def test_each_kind_of_node_gets_its_own_colour(card) -> None:
    markup = render_subgraph(card)

    used = {colour for colour in TYPE_COLOUR.values() if colour in markup}
    assert len(used) >= 2


def test_identifiers_are_escaped_not_injected() -> None:
    hostile = {
        "subject": "user:<script>alert(1)</script>",
        "object": "bucket:ok",
        "level": "admin",
        "relation": "HAS_PERMISSION",
        "subgraph": {
            "nodes": [
                {"id": "user:<script>alert(1)</script>", "type": "user", "hops": 0},
                {"id": "bucket:ok", "type": "bucket", "hops": 0},
            ],
            "edges": [
                {
                    "subject": "user:<script>alert(1)</script>",
                    "relation": "HAS_PERMISSION",
                    "object": "bucket:ok",
                    "level": "admin",
                    "importance": 0.5,
                }
            ],
        },
    }

    markup = render_subgraph(hostile)

    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup


def test_an_empty_neighbourhood_says_so_instead_of_drawing_nothing() -> None:
    markup = render_subgraph(
        {
            "subject": "user:a",
            "object": "bucket:b",
            "level": "read",
            "relation": "HAS_PERMISSION",
            "subgraph": {"nodes": [], "edges": []},
        }
    )

    assert "<text" in markup
    assert re.search(r"связ", markup, re.IGNORECASE)


def _crossings(markup: str) -> int:
    """How many pairs of drawn connections cross each other."""
    import re

    segments = [
        tuple(float(value) for value in match)
        for match in re.findall(
            r'<line[^>]*x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)"', markup
        )
    ]

    def side(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    total = 0
    for index, first in enumerate(segments):
        for second in segments[index + 1 :]:
            a = side(*first, second[0], second[1])
            b = side(*first, second[2], second[3])
            c = side(*second, first[0], first[1])
            d = side(*second, first[2], first[3])
            if a * b < 0 and c * d < 0:
                total += 1
    return total


def test_the_layout_untangles_what_it_can(card) -> None:
    """Ordering nodes by where their neighbours sit removes most crossings.

    Alphabetical order inside a column is arbitrary with respect to the edges, and
    on a dozen connections it produces a thicket. A couple of barycentre passes is
    the cheapest fix that keeps the layout deterministic.
    """
    from rga.explain.picture import render_subgraph as render

    tangled = render(card, edges=8, untangle=False)
    tidy = render(card, edges=8, untangle=True)

    assert _crossings(tidy) <= _crossings(tangled)
    assert _crossings(tidy) <= 4


def test_the_caption_says_what_is_drawn(card) -> None:
    markup = render_subgraph(card)

    assert "до изменения" in markup


def _boxes(markup: str) -> dict[str, float]:
    """The left edge of every drawn node, by its identifier."""
    import re

    found = {}
    for block in re.findall(r"<g>.*?</g>", markup, re.S):
        name = re.search(r"<title>([^<]+)</title>", block)
        x = re.search(r'<rect x="([-\d.]+)"', block)
        if name and x:
            found[name.group(1)] = float(x.group(1))
    return found


def _chain_card() -> dict:
    """A user in a group, the group holding a right on a bucket holding an object."""
    return {
        "subject": "user:alice",
        "object": "bucket:logs",
        "relation": "HAS_PERMISSION",
        "level": "admin",
        "subgraph": {
            "nodes": [
                {"id": "user:alice", "type": "user", "hops": 0},
                {"id": "group:devops", "type": "group", "hops": 1},
                {"id": "bucket:logs", "type": "bucket", "hops": 0},
                {"id": "object:logs/file-1", "type": "object", "hops": 2},
            ],
            "edges": [
                {"subject": "user:alice", "relation": "MEMBER_OF",
                 "object": "group:devops", "level": "none", "importance": 0.4},
                {"subject": "group:devops", "relation": "HAS_PERMISSION",
                 "object": "bucket:logs", "level": "admin", "importance": 0.3},
                {"subject": "bucket:logs", "relation": "PARENT_OF",
                 "object": "object:logs/file-1", "level": "none", "importance": 0.1},
            ],
        },
    }


def test_each_kind_of_node_gets_its_own_column() -> None:
    """Columns follow the path the authorization engine itself walks.

    Laying them out by distance from the change put a user and a group in the same
    column, so the line between them ran backwards and crossed everything else.
    A user is always left of a group, which is left of a bucket.
    """
    from rga.explain.picture import render_subgraph as render

    at = _boxes(render(_chain_card()))

    assert at["user:alice"] < at["group:devops"] < at["bucket:logs"] < at["object:logs/file-1"]


def test_a_chain_is_drawn_without_a_single_crossing() -> None:
    from rga.explain.picture import render_subgraph as render

    assert _crossings(render(_chain_card())) == 0


def test_empty_lanes_leave_no_gap() -> None:
    """A neighbourhood of users and buckets alone must not draw an empty middle."""
    from rga.explain.picture import render_subgraph as render

    card = _chain_card()
    card["subgraph"]["nodes"] = [
        node for node in card["subgraph"]["nodes"] if node["type"] in ("user", "bucket")
    ]
    card["subgraph"]["edges"] = [
        {"subject": "user:alice", "relation": "HAS_PERMISSION",
         "object": "bucket:logs", "level": "admin", "importance": 0.4}
    ]

    at = _boxes(render(card))
    gap = at["bucket:logs"] - at["user:alice"]

    from rga.explain.picture import _COLUMN

    assert gap == _COLUMN
