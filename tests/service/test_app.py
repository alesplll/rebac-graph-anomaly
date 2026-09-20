"""The routes the page talks to."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.supervised import SupervisedGnnScorer
from rga.service.app import create_app
from rga.service.config import load_service_config
from rga.service.triage import Decision, MemoryStore

FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3)


@pytest.fixture(scope="module")
def fitted():
    """The expensive half, paid for once."""
    dataset = build_dataset(load_dataset_config(Path("configs/generator/small-history.yaml")))
    scorer = SupervisedGnnScorer(seed=0, config=FAST)
    scorer.fit(build_candidates(dataset, Span.TRAIN))
    return scorer, load_service_config(Path("configs/service/synthetic.yaml"))


@pytest.fixture(scope="module")
def client(fitted):
    scorer, config = fitted
    return TestClient(create_app(config, scorer=scorer, store=MemoryStore()))


@pytest.fixture
def client_and_store(fitted):
    """A client with a journal of its own, thrown away with the test."""
    scorer, config = fitted
    store = MemoryStore()
    return TestClient(create_app(config, scorer=scorer, store=store)), store


def _decision_for(row: dict, outcome: str, note: str = "") -> Decision:
    return Decision(
        seq=0,
        incident=str(row["id"]),
        outcome=outcome,
        note=note,
        analyst="analyst",
        decided_at="2026-09-20T12:00:00+00:00",
        subject=str(row["subject"]),
        relation=str(row["relation"]),
        object=str(row["object"]),
        score=float(row["score"]),
    )


def test_status_describes_what_is_running(client) -> None:
    body = client.get("/api/status").json()

    assert body["source"] == "synthetic"
    assert body["capability_level"] >= 1
    assert body["scorer"] == "gnn_supervised"
    assert body["candidates"] > 0


def test_the_queue_comes_back_ranked(client) -> None:
    body = client.get("/api/incidents", params={"limit": 10}).json()

    assert len(body["incidents"]) == 10
    scores = [item["score"] for item in body["incidents"]]
    assert scores == sorted(scores, reverse=True)
    assert body["incidents"][0]["rank"] == 1


def test_the_queue_can_be_filtered_by_subject(client) -> None:
    everything = client.get("/api/incidents", params={"limit": 50}).json()["incidents"]
    subject = everything[0]["subject"]

    filtered = client.get("/api/incidents", params={"limit": 50, "subject": subject}).json()

    assert filtered["incidents"]
    assert {item["subject"] for item in filtered["incidents"]} == {subject}


def test_a_card_carries_its_grounds(client) -> None:
    listed = client.get("/api/incidents", params={"limit": 1}).json()["incidents"][0]

    card = client.get(f"/api/incidents/{listed['id']}").json()

    assert card["id"] == listed["id"]
    assert card["summary"]
    assert card["subgraph"]["nodes"]


def test_an_unknown_card_is_a_clean_404(client) -> None:
    assert client.get("/api/incidents/" + "0" * 16).status_code == 404


def test_a_node_neighbourhood_comes_back(client) -> None:
    listed = client.get("/api/incidents", params={"limit": 1}).json()["incidents"][0]

    body = client.get("/api/nodes", params={"id": listed["subject"]}).json()

    assert body["id"] == listed["subject"]
    assert body["subgraph"]["nodes"]


def test_an_unknown_node_is_a_clean_404(client) -> None:
    assert client.get("/api/nodes", params={"id": "user:nobody"}).status_code == 404


def test_refresh_reruns_the_analysis(client) -> None:
    before = client.get("/api/status").json()["refreshed_at"]

    assert client.post("/api/refresh").status_code == 200
    assert client.get("/api/status").json()["refreshed_at"] >= before


def test_the_reference_is_served(client) -> None:
    body = client.get("/api/reference").json()

    assert body["observations"]
    assert body["features"]
    assert {"edges", "features"} <= set(body["tables"])


def test_an_open_incident_says_so(client_and_store) -> None:
    client, _ = client_and_store
    assert client.get("/api/incidents").json()["incidents"][0]["state"] == "open"


def test_a_resolved_incident_leaves_the_queue(client_and_store) -> None:
    client, store = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]

    store.record([_decision_for(first, "dismissed")])

    remaining = client.get("/api/incidents").json()
    assert first["id"] not in {row["id"] for row in remaining["incidents"]}
    assert remaining["resolved"] == 1


def test_a_reopened_incident_comes_back(client_and_store) -> None:
    """A change of mind must put the change back in front of the analyst."""
    client, store = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]

    store.record([_decision_for(first, "revoked")])
    store.record([_decision_for(first, "reopened")])

    assert first["id"] in {row["id"] for row in client.get("/api/incidents").json()["incidents"]}


def test_resolved_incidents_can_be_asked_for(client_and_store) -> None:
    client, store = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]
    store.record([_decision_for(first, "dismissed")])

    rows = client.get("/api/incidents?state=resolved").json()["incidents"]

    assert [row["id"] for row in rows] == [first["id"]]
    assert rows[0]["state"] == "dismissed"


def test_the_counts_cover_the_whole_queue_not_the_page(client_and_store) -> None:
    """A limit shortens the page; it must not shorten the tally above it."""
    client, _ = client_and_store
    payload = client.get("/api/incidents?limit=3").json()

    assert len(payload["incidents"]) == 3
    assert payload["open"] > 3


def test_search_matches_subject_object_and_actor(client_and_store) -> None:
    client, _ = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]
    needle = first["subject"].split(":")[1][:8]

    rows = client.get(f"/api/incidents?q={needle}").json()["incidents"]

    assert rows
    assert all(
        needle in row["subject"] or needle in row["object"] or needle in (row["actor"] or "")
        for row in rows
    )


def test_search_ignores_case(client_and_store) -> None:
    client, _ = client_and_store
    assert client.get("/api/incidents?q=USER%3A").json()["incidents"]


def test_search_narrows_the_queue(client_and_store) -> None:
    """A search that returns everything would prove nothing."""
    client, _ = client_and_store
    everything = client.get("/api/incidents").json()["incidents"]
    needle = everything[0]["object"]

    rows = client.get(f"/api/incidents?q={needle}").json()["incidents"]

    assert 0 < len(rows) < len(everything)


def test_resolved_can_be_narrowed_to_one_outcome(client_and_store) -> None:
    client, store = client_and_store
    rows = client.get("/api/incidents").json()["incidents"]
    store.record([_decision_for(rows[0], "revoked")])
    store.record([_decision_for(rows[1], "dismissed")])

    only = client.get("/api/incidents?state=resolved&outcome=revoked").json()["incidents"]

    assert [row["id"] for row in only] == [rows[0]["id"]]


def test_no_grouping_by_default(client_and_store) -> None:
    client, _ = client_and_store
    assert client.get("/api/incidents").json()["groups"] == []


def test_grouping_by_subject_counts_and_lists(client_and_store) -> None:
    client, _ = client_and_store
    payload = client.get("/api/incidents?group=subject").json()

    groups = payload["groups"]
    assert groups
    assert sum(group["count"] for group in groups) == len(payload["incidents"])
    assert all(group["count"] == len(group["incidents"]) for group in groups)
    # A burst of grants from one subject is one of the planted patterns, so at
    # least one subject must collect more than a single change.
    assert max(group["count"] for group in groups) > 1


def test_groups_are_ordered_by_their_worst_change(client_and_store) -> None:
    client, _ = client_and_store
    groups = client.get("/api/incidents?group=subject").json()["groups"]

    scores = [group["top_score"] for group in groups]
    assert scores == sorted(scores, reverse=True)


def test_grouping_by_object_uses_the_object(client_and_store) -> None:
    client, _ = client_and_store
    payload = client.get("/api/incidents?group=object").json()

    keys = {group["key"] for group in payload["groups"]}
    assert keys == {row["object"] for row in payload["incidents"]}


def test_a_decision_is_recorded_and_removes_the_change(client_and_store) -> None:
    client, _ = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]

    answer = client.post(
        "/api/decisions",
        json={
            "incidents": [first["id"]],
            "outcome": "dismissed",
            "note": "плановые работы",
        },
    )

    assert answer.status_code == 200
    assert answer.json() == {"recorded": 1, "unknown": []}
    queue = client.get("/api/incidents").json()["incidents"]
    assert first["id"] not in {row["id"] for row in queue}


def test_one_call_decides_several_changes(client_and_store) -> None:
    """The whole point of grouping: close a subject's changes in one act."""
    client, _ = client_and_store
    rows = client.get("/api/incidents").json()["incidents"][:3]

    answer = client.post(
        "/api/decisions",
        json={"incidents": [row["id"] for row in rows], "outcome": "dismissed"},
    )

    assert answer.json()["recorded"] == 3
    assert len(client.get("/api/decisions").json()["decisions"]) == 3


def test_the_note_reaches_the_journal(client_and_store) -> None:
    client, _ = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]

    client.post(
        "/api/decisions",
        json={"incidents": [first["id"]], "outcome": "revoked", "note": "передано в SOC"},
    )

    assert client.get("/api/decisions").json()["decisions"][0]["note"] == "передано в SOC"


def test_an_unknown_outcome_is_refused(client_and_store) -> None:
    client, _ = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]

    answer = client.post("/api/decisions", json={"incidents": [first["id"]], "outcome": "maybe"})

    assert answer.status_code == 422


def test_an_empty_selection_is_refused(client_and_store) -> None:
    client, _ = client_and_store
    answer = client.post("/api/decisions", json={"incidents": [], "outcome": "revoked"})
    assert answer.status_code == 422


def test_a_change_outside_the_window_is_still_recorded(client_and_store) -> None:
    """The journal is self-contained; it does not need the incident to exist."""
    client, _ = client_and_store

    answer = client.post(
        "/api/decisions", json={"incidents": ["deadbeefdeadbeef"], "outcome": "revoked"}
    )

    assert answer.json() == {"recorded": 1, "unknown": ["deadbeefdeadbeef"]}


def test_the_card_carries_its_decision_and_its_history(client_and_store) -> None:
    client, _ = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]
    client.post("/api/decisions", json={"incidents": [first["id"]], "outcome": "revoked"})
    client.post("/api/decisions", json={"incidents": [first["id"]], "outcome": "reopened"})

    card = client.get(f"/api/incidents/{first['id']}").json()

    assert card["decision"]["outcome"] == "reopened"
    assert [entry["outcome"] for entry in card["history"]] == ["reopened", "revoked"]


def test_an_undecided_card_says_so(client_and_store) -> None:
    client, _ = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]

    card = client.get(f"/api/incidents/{first['id']}").json()

    assert card["decision"] is None
    assert card["history"] == []


def test_the_page_and_its_assets_are_always_revalidated(client) -> None:
    """A cached script against fresh markup is a dead page, and it happened.

    The browser kept an older app.js, which reached for an element the new markup
    no longer has, threw, and left the console inert. Nothing in the page said so —
    it simply sat there. These headers make the browser ask every time; the ETag
    keeps the answer cheap.
    """
    for path in ("/static/app.js", "/static/style.css"):
        headers = client.get(path).headers
        assert headers.get("cache-control") == "no-cache", path


def test_every_row_carries_a_readable_state(client_and_store) -> None:
    """The page shows the state as a word, so the service is the one that names it."""
    client, store = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]
    assert first["state"] == "open"
    assert first["state_title"] == "Открыто"

    store.record([_decision_for(first, "revoked", note="снял права")])
    row = client.get("/api/incidents?state=resolved").json()["incidents"][0]

    assert (row["state"], row["state_title"], row["note"]) == (
        "revoked",
        "Права отозваны",
        "снял права",
    )


def test_a_reopened_change_reads_as_open(client_and_store) -> None:
    client, store = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]
    store.record([_decision_for(first, "revoked")])
    store.record([_decision_for(first, "reopened")])

    row = next(
        item
        for item in client.get("/api/incidents").json()["incidents"]
        if item["id"] == first["id"]
    )

    assert row["state_title"] == "Открыто"


def test_the_page_names_its_assets_by_content(client) -> None:
    """A browser cannot serve a cached copy of a URL it has never seen.

    `Cache-Control` only reaches a browser that asks, and one holding a copy taken
    before the header existed does not ask: a response with no freshness information
    is cached heuristically. Putting the file's own fingerprint in the URL sidesteps
    the question — a changed file is a different address.
    """
    markup = client.get("/").text

    assert "/static/app.js?v=" in markup
    assert "/static/style.css?v=" in markup


def test_the_fingerprint_follows_the_file(client, tmp_path) -> None:
    from pathlib import Path as _Path

    before = client.get("/").text
    script = _Path("web/app.js")
    kept = script.read_text(encoding="utf-8")
    try:
        script.write_text(kept + "\n// a change\n", encoding="utf-8")
        after = client.get("/").text
    finally:
        script.write_text(kept, encoding="utf-8")

    assert before != after


def test_the_page_itself_is_never_stored(client) -> None:
    """The markup carries the fingerprints, so it is the one thing that must be fresh."""
    assert client.get("/").headers.get("cache-control") == "no-store"
