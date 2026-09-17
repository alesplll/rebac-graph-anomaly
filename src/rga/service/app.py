"""The HTTP surface: a queue, a card, a neighbourhood.

The service reads and never writes. It holds one analysis in memory and replaces it on
refresh, which is what the closing minute of the demonstration leans on: grant yourself
a right through the engine, press refresh, watch it arrive.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rga.domain.entities import entity_type
from rga.domain.relations import PermissionLevel, RelationType
from rga.explain.incident import build_incident
from rga.explain.structure import neighbourhood
from rga.service.analysis import Analysis, analyse
from rga.service.config import ServiceConfig

WEB = Path("web")


def create_app(config: ServiceConfig, *, scorer=None) -> FastAPI:
    """Build the application. `scorer` is injected by tests; otherwise it is loaded."""
    if scorer is None:
        from rga.artifacts import load_scorer

        scorer = load_scorer(config.model)

    app = FastAPI(title="Обзор изменений прав доступа", docs_url="/api/docs")
    state: dict[str, Analysis] = {"analysis": analyse(config, scorer)}

    def current() -> Analysis:
        return state["analysis"]

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
        since: int | None = None,
        relation: str | None = None,
        subject: str | None = None,
    ) -> dict[str, object]:
        analysis = current()
        rows = []
        for rank, position in enumerate(analysis.order.tolist(), start=1):
            key = analysis.candidates.keys[position]
            ts = int(analysis.candidates.ts[position])
            if since is not None and ts < since:
                continue
            if subject is not None and key[0] != subject:
                continue
            if relation is not None and RelationType(key[1]).name != relation:
                continue
            rows.append(
                build_incident(
                    scorer,
                    analysis.candidates,
                    position,
                    score=float(analysis.scores[position]),
                    rank=rank,
                    explain=False,
                ).as_dict()
            )
            if len(rows) >= limit:
                break
        return {"incidents": rows, "total": analysis.candidates.n_candidates}

    @app.get("/api/incidents/{incident}")
    def incident(incident: str) -> dict[str, object]:
        analysis = current()
        position = analysis.find(incident)
        if position is None:
            raise HTTPException(status_code=404, detail="no such change in the current window")
        rank = int(analysis.order.tolist().index(position)) + 1
        return build_incident(
            scorer,
            analysis.candidates,
            position,
            score=float(analysis.scores[position]),
            rank=rank,
        ).as_dict()

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

        @app.get("/")
        def page() -> FileResponse:
            return FileResponse(WEB / "index.html")

    return app
