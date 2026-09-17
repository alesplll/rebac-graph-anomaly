"""The neighbourhood of a change, drawn as SVG.

Rendered on the server rather than in the page, for one practical reason: here it is
testable. A layout written in the page's JavaScript fails silently — an exception
leaves an empty box and nothing says why — and this project has no browser in its
test suite to notice.

Laid out in columns by distance from the change instead of by a force simulation. At
a dozen nodes a deterministic layout reads better, never jitters between redraws, and
needs no library, which matters for a page that must open on a machine with no
network.
"""

from __future__ import annotations

from html import escape

#: A colour per kind of node, so a user is never mistaken for a bucket at a glance.
TYPE_COLOUR = {
    "user": "#2f6f4f",
    "group": "#1f5f8b",
    "bucket": "#7a5199",
    "object": "#8a6d1f",
}
_UNKNOWN_COLOUR = "#6b7079"

#: The change under review, drawn apart from everything else.
_CHANGE_COLOUR = "#c0392b"
#: A relationship that holds the score up, and one that pulls it down.
_HOLDS_UP = "#b4541f"
_PULLS_DOWN = "#1f5f8b"

_BOX = (186, 24)
_ROW = 38
_COLUMN = 250
_MARGIN = 24


def _short(entity_id: str, limit: int = 24) -> str:
    """The identifying part of an entity id, trimmed to fit a box."""
    _, separator, rest = entity_id.partition(":")
    name = rest if separator else entity_id
    return name if len(name) <= limit else f"{name[: limit - 1]}…"


def _empty(message: str) -> str:
    return (
        '<svg viewBox="0 0 520 60" role="img" xmlns="http://www.w3.org/2000/svg">'
        f'<text x="12" y="34" fill="#6b7079" font-size="13">{escape(message)}</text>'
        "</svg>"
    )


def _untangled(columns: dict[int, list[dict]], drawn: list[dict]) -> dict[int, list[dict]]:
    """Order each column by where its neighbours sit, to stop lines crossing.

    Alphabetical order inside a column is arbitrary with respect to the edges, and on
    a dozen connections it produces a thicket. Two barycentre passes — put each node
    at the average height of the nodes it connects to — remove most crossings and
    keep the layout deterministic, which a force simulation would not.
    """
    neighbours: dict[str, list[str]] = {}
    for edge in drawn:
        neighbours.setdefault(edge["subject"], []).append(edge["object"])
        neighbours.setdefault(edge["object"], []).append(edge["subject"])

    order = {
        node["id"]: index
        for column in columns.values()
        for index, node in enumerate(column)
    }

    for _ in range(2):
        for hops in sorted(columns):
            column = columns[hops]
            column.sort(
                key=lambda node: (
                    sum(order.get(near, 0) for near in neighbours.get(node["id"], []))
                    / max(len(neighbours.get(node["id"], [])), 1),
                    node["id"],
                )
            )
            for index, node in enumerate(column):
                order[node["id"]] = index
    return columns


def render_subgraph(card: dict, *, edges: int = 8, untangle: bool = True) -> str:
    """Draw the neighbourhood held in an incident payload.

    What is drawn is the graph **as it stood before the change**, plus the change
    itself as a dashed line. That is what makes the picture worth looking at: the
    difference between the two is exactly one edge, and everything else is the
    context the score was judged against.

    `edges` bounds how much is shown: the relationships that moved the score most,
    and only the nodes they touch. Everything within two hops of a busy user runs to
    dozens of nodes and reads as noise.
    """
    subgraph = card.get("subgraph") or {"nodes": [], "edges": []}
    if not subgraph["nodes"]:
        return _empty("Связей вокруг этого изменения не нашлось.")

    ranked = sorted(subgraph["edges"], key=lambda item: -abs(item.get("importance", 0.0)))
    drawn = ranked[:edges]

    keep = {card["subject"], card["object"]}
    for edge in drawn:
        keep.add(edge["subject"])
        keep.add(edge["object"])
    nodes = [node for node in subgraph["nodes"] if node["id"] in keep]
    if not nodes:
        return _empty("Связей вокруг этого изменения не нашлось.")

    columns: dict[int, list[dict]] = {}
    for node in sorted(nodes, key=lambda node: (node["hops"], node["id"])):
        columns.setdefault(int(node["hops"]), []).append(node)

    if untangle:
        columns = _untangled(columns, drawn)

    place: dict[str, tuple[int, int]] = {}
    for hops, column in columns.items():
        for index, node in enumerate(column):
            place[node["id"]] = (
                _MARGIN + hops * _COLUMN,
                _MARGIN + index * _ROW,
            )

    width = _MARGIN * 2 + (max(columns) + 1) * _COLUMN - (_COLUMN - _BOX[0])
    height = _MARGIN * 2 + max(len(column) for column in columns.values()) * _ROW
    strongest = max((abs(edge.get("importance", 0.0)) for edge in drawn), default=0.0) or 1.0

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'xmlns="http://www.w3.org/2000/svg" font-family="system-ui, sans-serif">',
        "<title>Граф до изменения; красным пунктиром — оцениваемое изменение</title>",
    ]

    for edge in drawn:
        start, finish = place.get(edge["subject"]), place.get(edge["object"])
        if start is None or finish is None:
            continue
        weight = abs(edge.get("importance", 0.0)) / strongest
        colour = _HOLDS_UP if edge.get("importance", 0.0) >= 0 else _PULLS_DOWN
        parts.append(
            f'<line x1="{start[0] + _BOX[0]}" y1="{start[1] + _BOX[1] // 2}" '
            f'x2="{finish[0]}" y2="{finish[1] + _BOX[1] // 2}" stroke="{colour}" '
            f'stroke-width="{1 + 4 * weight:.2f}" stroke-opacity="{0.3 + 0.6 * weight:.2f}" />'
        )

    start, finish = place.get(card["subject"]), place.get(card["object"])
    if start is not None and finish is not None:
        label = card.get("level") or ""
        caption = card["relation"] if label in ("", "none") else f'{card["relation"]} {label}'
        parts.append(
            f'<line class="change" x1="{start[0] + _BOX[0]}" y1="{start[1] + _BOX[1] // 2}" '
            f'x2="{finish[0]}" y2="{finish[1] + _BOX[1] // 2}" stroke="{_CHANGE_COLOUR}" '
            'stroke-width="3" stroke-dasharray="7 4" />'
        )
        parts.append(
            f'<text x="{(start[0] + _BOX[0] + finish[0]) // 2}" '
            f'y="{(start[1] + finish[1]) // 2 + 2}" fill="{_CHANGE_COLOUR}" '
            f'font-size="11" text-anchor="middle">{escape(caption)}</text>'
        )

    for node in nodes:
        x, y = place[node["id"]]
        colour = TYPE_COLOUR.get(node["type"], _UNKNOWN_COLOUR)
        centre = node["id"] in (card["subject"], card["object"])
        parts.append(
            f'<g><title>{escape(node["id"])}</title>'
            f'<rect x="{x}" y="{y}" width="{_BOX[0]}" height="{_BOX[1]}" rx="5" '
            f'fill="{colour}1a" stroke="{colour}" stroke-width="{2.5 if centre else 1}" />'
            f'<text x="{x + 9}" y="{y + 16}" font-size="11.5" fill="{colour}">'
            f'{escape(_short(node["id"]))}</text></g>'
        )

    parts.append("</svg>")
    return "".join(parts)
