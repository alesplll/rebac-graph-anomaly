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
