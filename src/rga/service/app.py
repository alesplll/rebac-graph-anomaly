"""The HTTP surface: a queue, a card, a neighbourhood, and the analyst's decisions.

The analysis is read-only and replaced wholesale on refresh, which is what the closing
minute of the demonstration leans on: grant yourself a right through the engine, press
refresh, watch it arrive. The one thing the service does write is the triage journal,
and that lives in its own module which knows nothing about scoring.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rga.domain.entities import entity_type
from rga.domain.relations import PermissionLevel, RelationType
from rga.explain.incident import build_incident, incident_id
from rga.explain.reference import reference
from rga.explain.structure import neighbourhood
from rga.service.analysis import Analysis, analyse
from rga.service.config import ServiceConfig
from rga.service.triage import OPEN_OUTCOME, OUTCOMES, TITLES, Decision, MemoryStore

WEB = Path("web")


def _versioned(page: Path) -> HTMLResponse:
    """The markup with each asset addressed by the fingerprint of its contents.

    `Cache-Control` only reaches a browser that asks, and one holding a copy taken
    before the header existed does not ask: a response carrying no freshness
    information is cached heuristically, for hours. That is not theory — it left a
    stale script running against fresh markup twice, with nothing on screen to say
    so. A fingerprint in the URL settles it: a changed file is a different address,
    and there is nothing cached under it.

    The markup itself is therefore the one thing that must never be stored, since it
    is what carries the fingerprints.
    """
    markup = page.read_text(encoding="utf-8")

    def stamp(match: re.Match[str]) -> str:
        name = match.group(1)
        asset = WEB / name
        if not asset.is_file():
            return match.group(0)
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()[:12]
        return f"/static/{name}?v={digest}"

    markup = re.sub(r"/static/([A-Za-z0-9_.-]+)", stamp, markup)
    return HTMLResponse(markup, headers={"Cache-Control": "no-store"})


class DecisionRequest(BaseModel):
    """What the analyst decided about a selection of changes."""

    incidents: list[str]
    outcome: str
    note: str = ""


def _grouped(rows: list[dict[str, object]], by: str) -> list[dict[str, object]]:
    """Collect the page into groups, worst group first.

    Built from the rows that survived the filters and the limit rather than from the
    whole queue: a heading promising eleven changes above a group showing three
    would be worse than no heading at all.
    """
    if by == "none":
        return []

    collected: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        collected.setdefault(str(row[by]), []).append(row)

    return sorted(
        (
            {
                "key": key,
                "count": len(members),
                "top_score": max(float(member["score"]) for member in members),
                "incidents": [str(member["id"]) for member in members],
            }
            for key, members in collected.items()
        ),
        key=lambda item: float(item["top_score"]),  # type: ignore[arg-type]
        reverse=True,
    )


def create_app(
    config: ServiceConfig, *, scorer=None, store: MemoryStore | None = None
) -> FastAPI:
    """Build the application. `scorer` and `store` are injected by tests."""
    if scorer is None:
        from rga.artifacts import load_scorer

        scorer = load_scorer(config.model)
    if store is None:
        store = MemoryStore()

    app = FastAPI(title="Обзор изменений прав доступа", docs_url="/api/docs")
    state: dict[str, Analysis] = {"analysis": analyse(config, scorer)}

    def current() -> Analysis:
        return state["analysis"]

    @app.get("/api/reference")
    def help_page() -> dict[str, object]:
        """What the columns mean, assembled from the texts the cards themselves use."""
        return reference()

    @app.get("/api/status")
    def status() -> dict[str, object]:
        analysis = current()
        return {
            "source": analysis.source,
            "capability_level": analysis.level,
            "scorer": scorer.name,
            "window": {"start": analysis.window[0], "end": analysis.window[1]},
            "candidates": analysis.candidates.n_candidates,
            "refreshed_at": analysis.refreshed_at,
        }

    @app.post("/api/refresh")
    def refresh() -> dict[str, object]:
        state["analysis"] = analyse(config, scorer)
        return status()

    @app.get("/api/incidents")
    def incidents(
        limit: int = Query(default=config.queue, ge=1, le=500),
        state: str = Query(default="open", pattern="^(open|resolved|all)$"),
        group: str = Query(default="none", pattern="^(none|subject|object)$"),
        q: str | None = None,
        outcome: str | None = None,
        since: int | None = None,
        relation: str | None = None,
        subject: str | None = None,
    ) -> dict[str, object]:
        analysis = current()
        # Read once per request rather than once per row: the journal is small, but
        # the queue is not, and a query per candidate would show in the page.
        decided = store.current()

        rows: list[dict[str, object]] = []
        open_count = 0
        resolved_count = 0

        for rank, position in enumerate(analysis.order.tolist(), start=1):
            key = analysis.candidates.keys[position]
            ts = int(analysis.candidates.ts[position])
            decision = decided.get(incident_id(key, ts))
            resolved = decision is not None and decision.outcome != OPEN_OUTCOME

            # Counted over the whole queue, so the tally above the page does not
            # shrink with the page.
            if resolved:
                resolved_count += 1
            else:
                open_count += 1

            if state == "open" and resolved:
                continue
            if state == "resolved" and not resolved:
                continue
            if since is not None and ts < since:
                continue
            if subject is not None and key[0] != subject:
                continue
            if relation is not None and RelationType(key[1]).name != relation:
                continue
            if outcome is not None and (decision is None or decision.outcome != outcome):
                continue
            if q:
                needle = q.casefold()
                actor = analysis.candidates.actors[position] or ""
                if not any(needle in part.casefold() for part in (key[0], key[2], actor)):
                    continue
            if len(rows) >= limit:
                continue

            row = build_incident(
                scorer,
                analysis.candidates,
                position,
                score=float(analysis.scores[position]),
                rank=rank,
                explain=False,
            ).as_dict()
            row["state"] = decision.outcome if resolved else "open"
            row["state_title"] = TITLES[row["state"]]
            row["note"] = decision.note if resolved else ""
            row["decided_at"] = decision.decided_at if decision is not None else None
            rows.append(row)

        return {
            "incidents": rows,
            "total": analysis.candidates.n_candidates,
            "open": open_count,
            "resolved": resolved_count,
            "groups": _grouped(rows, group),
        }

    @app.get("/api/incidents/{incident}")
    def incident(incident: str) -> dict[str, object]:
        analysis = current()
        position = analysis.find(incident)
        if position is None:
            raise HTTPException(status_code=404, detail="no such change in the current window")
        rank = int(analysis.order.tolist().index(position)) + 1
        card = build_incident(
            scorer,
            analysis.candidates,
            position,
            score=float(analysis.scores[position]),
            rank=rank,
        ).as_dict()

        decision = store.current().get(incident)
        resolved = decision is not None and decision.outcome != OPEN_OUTCOME
        card["state"] = decision.outcome if resolved else "open"
        card["state_title"] = TITLES[card["state"]]
        card["decision"] = decision.as_dict() if decision is not None else None
        card["history"] = [entry.as_dict() for entry in store.history(incident=incident)]
        return card

    @app.post("/api/decisions")
    def decide(request: DecisionRequest) -> dict[str, object]:
        """Record one judgement per selected change."""
        if request.outcome not in OUTCOMES:
            raise HTTPException(status_code=422, detail=f"unknown outcome: {request.outcome}")
        if not request.incidents:
            raise HTTPException(status_code=422, detail="no changes were selected")

        analysis = current()
        decided_at = datetime.now(UTC).isoformat(timespec="seconds")
        entries: list[Decision] = []
        unknown: list[str] = []

        for identifier in request.incidents:
            position = analysis.find(identifier)
            if position is None:
                # Recorded regardless: a decision stands on its own, and the window
                # may have moved since the analyst last loaded the page.
                unknown.append(identifier)
                subject, relation, target, score = "", "", "", 0.0
            else:
                key = analysis.candidates.keys[position]
                subject, relation, target = key[0], RelationType(key[1]).name, key[2]
                score = float(analysis.scores[position])

            entries.append(
                Decision(
                    seq=0,
                    incident=identifier,
                    outcome=request.outcome,
                    note=request.note,
                    analyst="analyst",
                    decided_at=decided_at,
                    subject=subject,
                    relation=relation,
                    object=target,
                    score=score,
                )
            )

        store.record(entries)
        return {"recorded": len(entries), "unknown": unknown}

    @app.get("/api/decisions")
    def decisions(limit: int = Query(default=200, ge=1, le=1000)) -> dict[str, object]:
        return {"decisions": [entry.as_dict() for entry in store.history(limit=limit)]}

    @app.get("/api/nodes")
    def node(id: str = Query(...)) -> dict[str, object]:
        analysis = current()
        graph = analysis.candidates.graph
        if graph is None or id not in graph.node_index:
            raise HTTPException(status_code=404, detail="no such node in the current graph")

        nodes = {id: 0}
        edges = []
        for edge in neighbourhood(graph, id, id, hops=1, cap=60):
            source = graph.node_ids[int(graph.edge_src[edge])]
            sink = graph.node_ids[int(graph.edge_dst[edge])]
            nodes.setdefault(source, 1)
            nodes.setdefault(sink, 1)
            edges.append(
                {
                    "subject": source,
                    "relation": RelationType(int(graph.edge_rel[edge])).name,
                    "object": sink,
                    "level": PermissionLevel(int(graph.edge_level[edge])).name.lower(),
                    "importance": 0.0,
                }
            )

        return {
            "id": id,
            "subgraph": {
                "nodes": [
                    {"id": name, "type": entity_type(name).name.lower(), "hops": hops}
                    for name, hops in nodes.items()
                ],
                "edges": edges,
            },
        }

    if WEB.is_dir():
        app.mount("/static", StaticFiles(directory=WEB), name="static")

        @app.middleware("http")
        async def always_revalidate(request, call_next):
            """Never let a browser run yesterday's script against today's markup."""
            response = await call_next(request)
            if request.url.path.startswith(("/static", "/reference")):
                response.headers["Cache-Control"] = "no-cache"
            return response

        @app.get("/")
        def page() -> HTMLResponse:
            return _versioned(WEB / "index.html")

        @app.get("/reference")
        def help_view() -> HTMLResponse:
            return _versioned(WEB / "reference.html")

    return app
