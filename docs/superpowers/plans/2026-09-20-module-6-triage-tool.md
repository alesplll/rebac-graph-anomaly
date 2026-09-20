# Модуль 6. Инструмент триажа: план

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Превратить страницу-список в рабочий инструмент: решения аналитика
переживают перезапуск, очередь фильтруется и группируется, разобранное уходит в
историю, рисунок показывает только объясняющие связи.

**Architecture:** Хранилище решений — отдельный модуль на стандартном `sqlite3`,
который ничего не знает про скоринг; анализ остаётся чистым; слой HTTP соединяет их
при чтении. Журнал только на дозапись, текущее состояние — последняя запись по
инциденту. Страница переписывается на плотную вёрстку с двумя темами.

**Tech Stack:** Python 3.12 через `uv`, FastAPI, стандартный `sqlite3`, HTML, CSS и
JavaScript без единой внешней зависимости.

**Spec:** `docs/superpowers/specs/2026-09-20-module-6-triage-tool-design.md`

## Global Constraints

- Python строго 3.12 через `uv`. Проверка: `uv run pytest -m "not integration and not gpu"`, `uv run ruff check .`.
- **Ни одной новой зависимости.** Хранилище — стандартный `sqlite3`, страница — без библиотек и без CDN.
- Комментарии и docstring в коде на английском. Сообщения коммитов короткие, на английском, без эмодзи, **без строк атрибуции ИИ**. После каждого коммита `git push origin main`. Веток не заводим.
- Пути только через `pathlib`. Перенос строки в файлах принудительно `\n`.
- **Решения аналитика никогда не попадают в модель.** `rga.nn` и `rga.features` не импортируют `rga.service`. Закрепляется тестом задачи 8.
- Система не выносит вердикт о компрометации: в текстах интерфейса решение принадлежит человеку, а не системе.
- Идентификаторы узлов приходят из чужой системы и попадают в разметку — экранировать везде, где они рисуются.
- Порядок шагов внутри задачи не сокращать: сначала падающий тест, потом подтверждение падения по ожидаемой причине, потом реализация, потом зелёный прогон, потом коммит.

---

### Task 1: Хранилище решений

**Files:**
- Create: `src/rga/service/triage.py`
- Test: `tests/service/test_triage.py`

**Interfaces:**
- Consumes: стандартные `sqlite3`, `pathlib`, `dataclasses`.
- Produces: `Decision` (frozen dataclass), `OUTCOMES`, `OPEN_OUTCOME`, `TriageStore` с методами `record`, `current`, `history`, `close`.

- [ ] **Step 1: Write the failing test**

Create `tests/service/test_triage.py`:

```python
"""The journal of what the analyst decided."""

from pathlib import Path

import pytest

from rga.service.triage import Decision, TriageStore


def _decision(incident: str, outcome: str, **overrides) -> Decision:
    fields = {
        "seq": 0,
        "incident": incident,
        "outcome": outcome,
        "note": "",
        "analyst": "analyst",
        "decided_at": "2026-09-20T12:00:00+00:00",
        "subject": "user:alice",
        "relation": "HAS_PERMISSION",
        "object": "bucket:logs",
        "score": 0.9,
    }
    fields.update(overrides)
    return Decision(**fields)


def test_a_recorded_decision_comes_back(tmp_path: Path) -> None:
    store = TriageStore(tmp_path / "t.db")
    store.record([_decision("abc", "confirmed", note="escalation")])

    current = store.current()

    assert set(current) == {"abc"}
    assert current["abc"].outcome == "confirmed"
    assert current["abc"].note == "escalation"


def test_the_latest_decision_wins(tmp_path: Path) -> None:
    store = TriageStore(tmp_path / "t.db")
    store.record([_decision("abc", "confirmed")])
    store.record([_decision("abc", "false_positive")])

    assert store.current()["abc"].outcome == "false_positive"


def test_history_keeps_every_decision_newest_first(tmp_path: Path) -> None:
    """The journal is append-only: nothing is overwritten and nothing is lost."""
    store = TriageStore(tmp_path / "t.db")
    store.record([_decision("abc", "confirmed")])
    store.record([_decision("abc", "reopened")])
    store.record([_decision("abc", "accepted_risk")])

    history = store.history(incident="abc")

    assert [entry.outcome for entry in history] == [
        "accepted_risk",
        "reopened",
        "confirmed",
    ]


def test_a_decision_survives_reopening_the_file(tmp_path: Path) -> None:
    path = tmp_path / "t.db"
    first = TriageStore(path)
    first.record([_decision("abc", "confirmed")])
    first.close()

    assert TriageStore(path).current()["abc"].outcome == "confirmed"


def test_a_batch_records_one_row_per_incident(tmp_path: Path) -> None:
    store = TriageStore(tmp_path / "t.db")

    store.record(
        [_decision("a", "accepted_risk"), _decision("b", "accepted_risk")]
    )

    assert set(store.current()) == {"a", "b"}
    assert len(store.history()) == 2


def test_an_unknown_outcome_is_refused(tmp_path: Path) -> None:
    store = TriageStore(tmp_path / "t.db")
    with pytest.raises(ValueError, match="outcome"):
        store.record([_decision("abc", "probably-fine")])


def test_the_decision_carries_what_it_was_about(tmp_path: Path) -> None:
    """The window moves; the history has to stay readable without the incident."""
    store = TriageStore(tmp_path / "t.db")
    store.record(
        [_decision("abc", "confirmed", subject="user:bob", object="bucket:pay", score=0.77)]
    )

    entry = store.history()[0]

    assert (entry.subject, entry.object, entry.score) == ("user:bob", "bucket:pay", 0.77)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/service/test_triage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rga.service.triage'`.

- [ ] **Step 3: Write the implementation**

Create `src/rga/service/triage.py`:

```python
"""What the analyst decided, and when.

An append-only journal. A decision is never edited and never deleted: a change of
mind adds a row, and the current state of an incident is its latest row. That is
what makes the history worth having — the alternative, a mutable flag, answers "is
this closed" and nothing else.

Each row carries a copy of the change it was about. The evaluation window moves and
the source can be swapped, so an incident that was decided on yesterday may not
exist in today's analysis; the history has to stay readable regardless.

Nothing here ever reaches the model. These are human judgements about the very
changes the model ranks, and feeding them back would turn every measured number
into self-confirmation.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: Outcomes a decision may carry. `reopened` puts the incident back in the queue.
OUTCOMES: tuple[str, ...] = ("confirmed", "false_positive", "accepted_risk", "reopened")
#: The outcome that leaves an incident in the queue rather than taking it out.
OPEN_OUTCOME = "reopened"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    incident   TEXT    NOT NULL,
    outcome    TEXT    NOT NULL,
    note       TEXT    NOT NULL DEFAULT '',
    analyst    TEXT    NOT NULL DEFAULT 'analyst',
    decided_at TEXT    NOT NULL,
    subject    TEXT    NOT NULL,
    relation   TEXT    NOT NULL,
    object     TEXT    NOT NULL,
    score      REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS decisions_incident ON decisions (incident);
CREATE INDEX IF NOT EXISTS decisions_seq ON decisions (seq DESC);
"""

_COLUMNS = (
    "seq, incident, outcome, note, analyst, decided_at, subject, relation, object, score"
)


@dataclass(frozen=True)
class Decision:
    """One recorded judgement about one change."""

    seq: int
    incident: str
    outcome: str
    note: str
    analyst: str
    decided_at: str
    subject: str
    relation: str
    object: str
    score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "incident": self.incident,
            "outcome": self.outcome,
            "note": self.note,
            "analyst": self.analyst,
            "decided_at": self.decided_at,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "score": self.score,
        }


class TriageStore:
    """The decision journal, kept in one SQLite file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The service is single-process but FastAPI answers on a thread pool, so the
        # connection is shared across threads and guarded by SQLite's own lock.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def record(self, entries: Sequence[Decision]) -> None:
        """Append one row per decision."""
        for entry in entries:
            if entry.outcome not in OUTCOMES:
                raise ValueError(f"unknown outcome: {entry.outcome!r}")
        self._db.executemany(
            "INSERT INTO decisions"
            " (incident, outcome, note, analyst, decided_at,"
            "  subject, relation, object, score)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    entry.incident,
                    entry.outcome,
                    entry.note,
                    entry.analyst,
                    entry.decided_at,
                    entry.subject,
                    entry.relation,
                    entry.object,
                    entry.score,
                )
                for entry in entries
            ],
        )
        self._db.commit()

    def current(self) -> dict[str, Decision]:
        """The latest decision for every incident that has one."""
        rows = self._db.execute(
            f"SELECT {_COLUMNS} FROM decisions"
            " WHERE seq IN (SELECT MAX(seq) FROM decisions GROUP BY incident)"
        ).fetchall()
        return {row["incident"]: _decision(row) for row in rows}

    def history(self, *, incident: str | None = None, limit: int = 200) -> tuple[Decision, ...]:
        """Recorded decisions, newest first."""
        if incident is None:
            rows = self._db.execute(
                f"SELECT {_COLUMNS} FROM decisions ORDER BY seq DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self._db.execute(
                f"SELECT {_COLUMNS} FROM decisions WHERE incident = ?"
                " ORDER BY seq DESC LIMIT ?",
                (incident, limit),
            ).fetchall()
        return tuple(_decision(row) for row in rows)

    def close(self) -> None:
        self._db.close()


def _decision(row: sqlite3.Row) -> Decision:
    return Decision(
        seq=int(row["seq"]),
        incident=str(row["incident"]),
        outcome=str(row["outcome"]),
        note=str(row["note"]),
        analyst=str(row["analyst"]),
        decided_at=str(row["decided_at"]),
        subject=str(row["subject"]),
        relation=str(row["relation"]),
        object=str(row["object"]),
        score=float(row["score"]),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/service/test_triage.py -v`
Expected: PASS, все семь.

- [ ] **Step 5: Commit**

```bash
uv run ruff check .
git add src/rga/service/triage.py tests/service/test_triage.py
git commit -m "feat: keep an append-only journal of triage decisions"
git push origin main
```

---

### Task 2: Путь к хранилищу в конфигурации

**Files:**
- Modify: `src/rga/service/config.py`
- Modify: `configs/service/synthetic.yaml`, `configs/service/opens3.yaml`, `configs/service/full.yaml`
- Test: `tests/service/test_analysis.py` (там уже читается конфигурация) или новый тест в нём же

**Interfaces:**
- Produces: `ServiceConfig.store: Path`, со значением по умолчанию `var/triage-<name>.db`, где `<name>` — имя файла конфигурации без расширения.

- [ ] **Step 1: Write the failing test**

Дописать в `tests/service/test_analysis.py`:

```python
def test_the_store_path_defaults_to_the_configuration_name(tmp_path) -> None:
    """Two demonstrations must not share a decision journal."""
    path = tmp_path / "demo.yaml"
    path.write_text(
        "source:\n  kind: synthetic\n  dataset: configs/generator/small.yaml\n"
        "model: artifacts/gnn-supervised\n",
        encoding="utf-8",
    )

    config = load_service_config(path)

    assert config.store == Path("var/triage-demo.db")


def test_the_store_path_can_be_set_explicitly(tmp_path) -> None:
    path = tmp_path / "demo.yaml"
    path.write_text(
        "source:\n  kind: synthetic\n  dataset: configs/generator/small.yaml\n"
        "model: artifacts/gnn-supervised\nstore: var/custom.db\n",
        encoding="utf-8",
    )

    assert load_service_config(path).store == Path("var/custom.db")
```

Импортировать `load_service_config` и `Path`, если их там ещё нет.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/service/test_analysis.py -v -k store`
Expected: FAIL with `AttributeError: 'ServiceConfig' object has no attribute 'store'`.

- [ ] **Step 3: Write the implementation**

В `src/rga/service/config.py` добавить поле в датакласс, после `mapping`:

```python
    #: Where the triage journal lives. Separate per configuration, so the synthetic
    #: demonstration and the live one never share decisions.
    store: Path = Path("var/triage.db")
```

и в `load_service_config`, перед `return`:

```python
    default_store = Path("var") / f"triage-{path.stem}.db"
```

добавив в конструктор `store=Path(document["store"]) if "store" in document else default_store,`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/service/ -v`
Expected: PASS.

- [ ] **Step 5: Ignore the journal directory**

Добавить строку `var/` в `.gitignore`: журнал — рабочее состояние, а не исходный
текст, и в репозиторий он не попадает по той же причине, что и `experiments/runs/`.

- [ ] **Step 6: Commit**

```bash
uv run ruff check .
git add src/rga/service/config.py tests/service/test_analysis.py .gitignore
git commit -m "feat: give each service configuration its own decision journal"
git push origin main
```

---

### Task 3: Состояние инцидента в очереди

**Files:**
- Modify: `src/rga/service/app.py`
- Test: `tests/service/test_app.py`

**Interfaces:**
- Consumes: `TriageStore`, `OPEN_OUTCOME` из задачи 1; `ServiceConfig.store` из задачи 2.
- Produces: `create_app(config, *, scorer=None, store=None)` — `store` внедряется тестами; `GET /api/incidents` принимает `state` и возвращает `open` и `resolved` в ответе, а каждый элемент получает поля `state` и `decided_at`.

- [ ] **Step 1: Write the failing test**

Дописать в `tests/service/test_app.py` (взять оттуда же существующий приём создания клиента и подменного скорера):

```python
def test_a_resolved_incident_leaves_the_queue(client_and_store) -> None:
    client, store = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]

    store.record([_decision_for(first, "false_positive")])

    remaining = client.get("/api/incidents").json()
    assert first["id"] not in {row["id"] for row in remaining["incidents"]}
    assert remaining["resolved"] == 1


def test_a_reopened_incident_comes_back(client_and_store) -> None:
    client, store = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]

    store.record([_decision_for(first, "confirmed")])
    store.record([_decision_for(first, "reopened")])

    assert first["id"] in {
        row["id"] for row in client.get("/api/incidents").json()["incidents"]
    }


def test_resolved_incidents_can_be_asked_for(client_and_store) -> None:
    client, store = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]
    store.record([_decision_for(first, "accepted_risk")])

    rows = client.get("/api/incidents?state=resolved").json()["incidents"]

    assert [row["id"] for row in rows] == [first["id"]]
    assert rows[0]["state"] == "accepted_risk"


def test_an_open_incident_says_so(client_and_store) -> None:
    client, _ = client_and_store
    assert client.get("/api/incidents").json()["incidents"][0]["state"] == "open"
```

Вспомогательная функция и фикстура в том же файле:

```python
@pytest.fixture
def client_and_store(tmp_path):
    """A client whose service writes its decisions to a throwaway journal."""
    from fastapi.testclient import TestClient

    from rga.service.triage import TriageStore

    store = TriageStore(tmp_path / "t.db")
    app = create_app(CONFIG, scorer=SCORER, store=store)
    return TestClient(app), store


def _decision_for(row: dict, outcome: str) -> "Decision":
    from rga.service.triage import Decision

    return Decision(
        seq=0,
        incident=row["id"],
        outcome=outcome,
        note="",
        analyst="analyst",
        decided_at="2026-09-20T12:00:00+00:00",
        subject=row["subject"],
        relation=row["relation"],
        object=row["object"],
        score=float(row["score"]),
    )
```

`CONFIG` и `SCORER` — то, чем уже пользуется существующий тест в этом файле; если
они там устроены иначе, взять их способ, а не заводить второй.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/service/test_app.py -v -k "resolved or reopened or open"`
Expected: FAIL with `TypeError: create_app() got an unexpected keyword argument 'store'`.

- [ ] **Step 3: Write the implementation**

В `src/rga/service/app.py`:

```python
from rga.service.triage import OPEN_OUTCOME, TriageStore


def create_app(config: ServiceConfig, *, scorer=None, store: TriageStore | None = None) -> FastAPI:
    """Build the application. `scorer` and `store` are injected by tests."""
    if scorer is None:
        from rga.artifacts import load_scorer

        scorer = load_scorer(config.model)
    if store is None:
        store = TriageStore(config.store)
```

Заменить тело `incidents` на следующее. Решения читаются один раз на запрос, а не
на строку.

```python
    @app.get("/api/incidents")
    def incidents(
        limit: int = Query(default=config.queue, ge=1, le=500),
        state: str = Query(default="open", pattern="^(open|resolved|all)$"),
        since: int | None = None,
        relation: str | None = None,
        subject: str | None = None,
    ) -> dict[str, object]:
        analysis = current()
        decided = store.current()
        rows: list[dict[str, object]] = []
        open_count = resolved_count = 0

        for rank, position in enumerate(analysis.order.tolist(), start=1):
            key = analysis.candidates.keys[position]
            ts = int(analysis.candidates.ts[position])
            identifier = incident_id(key, ts)
            decision = decided.get(identifier)
            resolved = decision is not None and decision.outcome != OPEN_OUTCOME
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
            row["decided_at"] = decision.decided_at if decision is not None else None
            rows.append(row)

        return {
            "incidents": rows,
            "total": analysis.candidates.n_candidates,
            "open": open_count,
            "resolved": resolved_count,
        }
```

Импортировать `incident_id` из `rga.explain.incident`.

Обратите внимание: счётчики считаются по всей очереди, поэтому `continue` по
`limit` стоит после них, а не `break`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/service/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check .
git add src/rga/service/app.py tests/service/test_app.py
git commit -m "feat: take resolved changes out of the queue"
git push origin main
```

---

### Task 4: Поиск подстрокой и фильтр по исходу

**Files:**
- Modify: `src/rga/service/app.py`
- Test: `tests/service/test_app.py`

**Interfaces:**
- Produces: `GET /api/incidents` принимает `q` и `outcome`.

- [ ] **Step 1: Write the failing test**

```python
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


def test_search_is_case_insensitive(client_and_store) -> None:
    client, _ = client_and_store
    rows = client.get("/api/incidents?q=USER:").json()["incidents"]
    assert rows


def test_resolved_can_be_narrowed_to_one_outcome(client_and_store) -> None:
    client, store = client_and_store
    rows = client.get("/api/incidents").json()["incidents"]
    store.record([_decision_for(rows[0], "confirmed")])
    store.record([_decision_for(rows[1], "false_positive")])

    only = client.get("/api/incidents?state=resolved&outcome=confirmed").json()["incidents"]

    assert [row["id"] for row in only] == [rows[0]["id"]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/service/test_app.py -v -k "search or outcome"`
Expected: FAIL — `q` и `outcome` не фильтруют, поэтому третий тест вернёт две
строки вместо одной.

- [ ] **Step 3: Write the implementation**

Добавить параметры в сигнатуру `incidents`:

```python
        q: str | None = None,
        outcome: str | None = None,
```

и после проверки `relation` — два условия:

```python
            if outcome is not None and (decision is None or decision.outcome != outcome):
                continue
            if q:
                needle = q.casefold()
                haystack = (key[0], key[2], analysis.candidates.actors[position] or "")
                if not any(needle in str(part).casefold() for part in haystack):
                    continue
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/service/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check .
git add src/rga/service/app.py tests/service/test_app.py
git commit -m "feat: search the queue and narrow it by outcome"
git push origin main
```

---

### Task 5: Группировка

**Files:**
- Modify: `src/rga/service/app.py`
- Test: `tests/service/test_app.py`

**Interfaces:**
- Produces: `GET /api/incidents?group=none|subject|object` добавляет в ответ `groups` — список `{"key", "count", "top_score", "incidents": [id, ...]}`, отсортированный по убыванию наибольшей оценки.

- [ ] **Step 1: Write the failing test**

```python
def test_grouping_by_subject_counts_and_lists(client_and_store) -> None:
    client, _ = client_and_store
    payload = client.get("/api/incidents?group=subject").json()

    groups = payload["groups"]
    assert groups
    assert sum(group["count"] for group in groups) == len(payload["incidents"])
    assert all(group["count"] == len(group["incidents"]) for group in groups)
    # A busy subject should collect more than one change.
    assert max(group["count"] for group in groups) > 1


def test_groups_are_ordered_by_their_worst_change(client_and_store) -> None:
    client, _ = client_and_store
    groups = client.get("/api/incidents?group=subject").json()["groups"]

    scores = [group["top_score"] for group in groups]
    assert scores == sorted(scores, reverse=True)


def test_no_grouping_by_default(client_and_store) -> None:
    client, _ = client_and_store
    assert client.get("/api/incidents").json()["groups"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/service/test_app.py -v -k group`
Expected: FAIL with `KeyError: 'groups'`.

- [ ] **Step 3: Write the implementation**

Добавить параметр:

```python
        group: str = Query(default="none", pattern="^(none|subject|object)$"),
```

и перед `return` собрать группы из уже отобранных строк:

```python
        groups: list[dict[str, object]] = []
        if group != "none":
            collected: dict[str, list[dict[str, object]]] = {}
            for row in rows:
                collected.setdefault(str(row[group]), []).append(row)
            groups = sorted(
                (
                    {
                        "key": key,
                        "count": len(members),
                        "top_score": max(float(member["score"]) for member in members),
                        "incidents": [str(member["id"]) for member in members],
                    }
                    for key, members in collected.items()
                ),
                key=lambda item: float(item["top_score"]),
                reverse=True,
            )
```

и вернуть `"groups": groups` в теле ответа.

Группы строятся по тем строкам, которые прошли фильтры и ограничение — иначе
счётчик в заголовке группы обещал бы больше, чем в ней видно.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/service/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check .
git add src/rga/service/app.py tests/service/test_app.py
git commit -m "feat: group the queue by subject or object"
git push origin main
```

---

### Task 6: Запись решений и история

**Files:**
- Modify: `src/rga/service/app.py`
- Test: `tests/service/test_app.py`

**Interfaces:**
- Produces: `POST /api/decisions` принимает `{"incidents": [...], "outcome": "...", "note": "..."}` и отдаёт `{"recorded": N, "unknown": [...]}`; `GET /api/decisions?limit=200` отдаёт `{"decisions": [...]}`; `GET /api/incidents/{id}` дополняется полями `decision` и `history`.

- [ ] **Step 1: Write the failing test**

```python
def test_a_decision_is_recorded_and_removes_the_change(client_and_store) -> None:
    client, _ = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]

    answer = client.post(
        "/api/decisions",
        json={"incidents": [first["id"]], "outcome": "accepted_risk", "note": "плановые работы"},
    )

    assert answer.status_code == 200
    assert answer.json() == {"recorded": 1, "unknown": []}
    assert first["id"] not in {
        row["id"] for row in client.get("/api/incidents").json()["incidents"]
    }


def test_one_call_decides_several_changes(client_and_store) -> None:
    client, _ = client_and_store
    rows = client.get("/api/incidents").json()["incidents"][:3]

    answer = client.post(
        "/api/decisions",
        json={"incidents": [row["id"] for row in rows], "outcome": "false_positive"},
    )

    assert answer.json()["recorded"] == 3
    assert len(client.get("/api/decisions").json()["decisions"]) == 3


def test_an_unknown_outcome_is_refused(client_and_store) -> None:
    client, _ = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]

    answer = client.post(
        "/api/decisions", json={"incidents": [first["id"]], "outcome": "maybe"}
    )

    assert answer.status_code == 422


def test_an_empty_selection_is_refused(client_and_store) -> None:
    client, _ = client_and_store
    answer = client.post("/api/decisions", json={"incidents": [], "outcome": "confirmed"})
    assert answer.status_code == 422


def test_a_change_outside_the_window_is_still_recorded(client_and_store) -> None:
    """The journal is self-contained; it does not need the incident to exist."""
    client, _ = client_and_store

    answer = client.post(
        "/api/decisions", json={"incidents": ["deadbeefdeadbeef"], "outcome": "confirmed"}
    )

    assert answer.json() == {"recorded": 1, "unknown": ["deadbeefdeadbeef"]}


def test_the_card_carries_its_decision_and_its_history(client_and_store) -> None:
    client, _ = client_and_store
    first = client.get("/api/incidents").json()["incidents"][0]
    client.post("/api/decisions", json={"incidents": [first["id"]], "outcome": "confirmed"})
    client.post("/api/decisions", json={"incidents": [first["id"]], "outcome": "reopened"})

    card = client.get(f"/api/incidents/{first['id']}").json()

    assert card["decision"]["outcome"] == "reopened"
    assert [entry["outcome"] for entry in card["history"]] == ["reopened", "confirmed"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/service/test_app.py -v -k "decision or decided or card"`
Expected: FAIL — `POST /api/decisions` отдаёт 405, ручки нет.

- [ ] **Step 3: Write the implementation**

В `src/rga/service/app.py` добавить модель запроса и две ручки. Импортировать
`datetime`, `UTC`, `BaseModel` из `pydantic`, `Decision` и `OUTCOMES`.

```python
class DecisionRequest(BaseModel):
    """What the analyst decided about a selection of changes."""

    incidents: list[str]
    outcome: str
    note: str = ""


    @app.post("/api/decisions")
    def decide(request: DecisionRequest) -> dict[str, object]:
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
                # Recorded anyway: a decision is self-contained, and the window may
                # have moved since the analyst opened the page.
                unknown.append(identifier)
                subject, relation, target, score = "", "", "", 0.0
            else:
                key = analysis.candidates.keys[position]
                subject = key[0]
                relation = RelationType(key[1]).name
                target = key[2]
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
```

В ручке `incident` добавить перед `return`:

```python
        card = build_incident(...).as_dict()
        decision = store.current().get(incident)
        card["decision"] = decision.as_dict() if decision is not None else None
        card["history"] = [entry.as_dict() for entry in store.history(incident=incident)]
        return card
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/service/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check .
git add src/rga/service/app.py tests/service/test_app.py
git commit -m "feat: record analyst decisions and show their history"
git push origin main
```

---

### Task 7: Рисунок только по объясняющим связям

**Files:**
- Modify: `src/rga/explain/incident.py`
- Test: `tests/explain/test_picture.py`, `tests/explain/test_incident.py`

**Interfaces:**
- Produces: `DRAWN_EDGES = 6`; `_subgraph` рисует не больше `DRAWN_EDGES` рёбер с наибольшим модулем вклада плюс само изменение, если оно есть в графе.

- [ ] **Step 1: Write the failing test**

Дописать в `tests/explain/test_incident.py`:

```python
def test_only_the_explaining_edges_are_drawn(fitted) -> None:
    """Sixty edges of a two-hop ball is a thicket; the score leans on a handful."""
    from rga.explain.incident import DRAWN_EDGES

    card = _card(fitted)

    drawn = card.subgraph["edges"]
    assert len(drawn) <= DRAWN_EDGES + 1


def test_the_drawn_edges_are_the_ones_that_matter(fitted) -> None:
    from rga.explain.incident import DRAWN_EDGES

    card = _card(fitted)

    best = sorted((abs(item.importance) for item in card.edges), reverse=True)
    threshold = best[DRAWN_EDGES - 1] if len(best) >= DRAWN_EDGES else 0.0
    weights = [abs(edge["importance"]) for edge in card.subgraph["edges"]]
    # Everything drawn either carries weight at or above the cut, or is the change.
    assert all(weight >= threshold or weight == 0.0 for weight in weights)
```

`_card(fitted)` — вызов `build_incident` тем же способом, каким это уже делают
соседние тесты в файле; если там есть фикстура, пользоваться ею.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/explain/test_incident.py -v -k drawn`
Expected: FAIL — рисуется до 60 рёбер, `len(drawn) <= 7` не выполняется.

- [ ] **Step 3: Write the implementation**

В `src/rga/explain/incident.py` добавить рядом с прочими константами:

```python
#: How many neighbouring relationships the drawing shows.
#:
#: The two-hop ball around a change runs to dozens of edges and draws as a thicket.
#: The score does not lean on dozens: it leans on a handful, and the picture exists
#: to show which. The rest stays in the contributions table, where a number reads
#: better than a line.
DRAWN_EDGES = 6
```

В `_subgraph` заменить построение `drawn`: сначала отобрать позиции, потом рисовать.

```python
    importance = {(item.subject, item.relation, item.object): item.importance for item in edges}
    positions = neighbourhood(graph, subject, target, cap=cap)

    def weight(edge: int) -> float:
        source = graph.node_ids[int(graph.edge_src[edge])]
        sink = graph.node_ids[int(graph.edge_dst[edge])]
        relation = RelationType(int(graph.edge_rel[edge])).name
        if (source, sink) == (subject, target):
            # The change itself is always drawn: it is what the card is about.
            return float("inf")
        return abs(float(importance.get((source, relation, sink), 0.0)))

    chosen = sorted(positions.tolist(), key=weight, reverse=True)[: DRAWN_EDGES + 1]

    hops = {name: 0 for name in (subject, target) if name in graph.node_index}
    drawn = []
    for edge in chosen:
        ...
```

Остальное тело метода не меняется.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/explain/ -v`
Expected: PASS. Тест на пересечения линий из модуля 4 должен пройти тем более:
рёбер стало меньше.

- [ ] **Step 5: Look at the result**

Run: `uv run rga serve --config configs/service/synthetic.yaml` и открыть карточку.
Expected: рисунок из нескольких связей вместо паутины. Если он всё ещё нечитаем —
это находка, и разбирать её надо по `superpowers:systematic-debugging`, а не
уменьшать `DRAWN_EDGES` наугад.

- [ ] **Step 6: Commit**

```bash
uv run ruff check .
git add src/rga/explain/incident.py tests/explain/test_incident.py
git commit -m "feat: draw only the relationships that hold the score up"
git push origin main
```

---

### Task 8: Проверка разделения слоёв

**Files:**
- Create: `tests/test_layering.py`

**Interfaces:**
- Consumes: стандартные `ast`, `pathlib`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_layering.py`:

```python
"""The model must never learn from what the analyst decided.

A judgement recorded in the triage journal is a human opinion about the very changes
the model ranks. Any path by which it reaches training turns every measured number
into self-confirmation, so the absence of that path is checked mechanically rather
than promised in prose.
"""

import ast
from pathlib import Path

import pytest

MODEL_PACKAGES = ("src/rga/nn", "src/rga/features", "src/rga/baselines")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


@pytest.mark.parametrize("package", MODEL_PACKAGES)
def test_the_model_never_imports_the_service(package: str) -> None:
    for path in sorted(Path(package).rglob("*.py")):
        offending = {name for name in _imports(path) if name.startswith("rga.service")}
        assert not offending, f"{path} imports {offending}"


def test_the_guard_would_notice(tmp_path: Path) -> None:
    """A test that cannot fail is not a test."""
    path = tmp_path / "leaky.py"
    path.write_text("from rga.service.triage import TriageStore\n", encoding="utf-8")

    assert {name for name in _imports(path) if name.startswith("rga.service")}
```

- [ ] **Step 2: Run test to verify it passes for the right reason**

Run: `uv run pytest tests/test_layering.py -v`
Expected: PASS. Это охранный тест: он обязан быть зелёным с самого начала, и его
ценность в том, что он покраснеет, если кто-нибудь проведёт такой импорт. Второй
тест доказывает, что проверка не пустая.

- [ ] **Step 3: Commit**

```bash
uv run ruff check .
git add tests/test_layering.py
git commit -m "test: keep triage decisions out of the model"
git push origin main
```

---

### Task 9: Страница — каркас, темы, фильтры

**Files:**
- Modify: `web/index.html`, `web/style.css`, `web/app.js`
- Test: `tests/service/test_page.py`

**Interfaces:**
- Consumes: ручки задач 3–6.
- Produces: разметка с `#filters`, `#queue`, `#selection`, `#detail`, `#history`, `[data-theme]` на корне и кнопкой `#theme`.

- [ ] **Step 1: Write the failing test**

Дописать в `tests/service/test_page.py`:

```python
def test_the_page_carries_the_filter_bar_and_the_theme_switch(client) -> None:
    markup = client.get("/").text

    for marker in ('id="filters"', 'id="queue"', 'id="selection"', 'id="theme"'):
        assert marker in markup


def test_the_stylesheet_defines_both_themes(client) -> None:
    css = client.get("/static/style.css").text

    assert ":root" in css
    assert '[data-theme="dark"]' in css
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/service/test_page.py -v`
Expected: FAIL — нынешняя разметка этих узлов не содержит.

- [ ] **Step 3: Write the markup**

`web/index.html` — каркас. Ничего не грузится извне, стиль и скрипт лежат рядом.

```html
<!doctype html>
<html lang="ru" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Обзор изменений прав доступа</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header id="top">
    <div class="brand">Изменения прав доступа</div>
    <div id="status" class="status"></div>
    <nav class="tabs">
      <button data-tab="queue" class="active">Очередь</button>
      <button data-tab="history">История</button>
    </nav>
    <button id="refresh" title="Перечитать источник">Обновить</button>
    <button id="theme" title="Светлая или тёмная тема">◐</button>
  </header>

  <section id="filters">
    <input id="search" type="search" placeholder="Поиск по субъекту, объекту, инициатору">
    <select id="state">
      <option value="open">Открытые</option>
      <option value="resolved">Разобранные</option>
      <option value="all">Все</option>
    </select>
    <select id="outcome">
      <option value="">Любой исход</option>
      <option value="confirmed">Подтверждено</option>
      <option value="false_positive">Ложное срабатывание</option>
      <option value="accepted_risk">Принятый риск</option>
    </select>
    <select id="group">
      <option value="none">Без группировки</option>
      <option value="subject">По субъекту</option>
      <option value="object">По объекту</option>
    </select>
    <span id="counts" class="counts"></span>
  </section>

  <main>
    <section id="queue" class="panel"></section>
    <aside id="detail" class="panel"></aside>
  </main>

  <section id="history" class="panel" hidden></section>

  <div id="selection" hidden>
    <span id="selected"></span>
    <input id="note" type="text" placeholder="Заметка к решению">
    <button data-outcome="confirmed">Подтверждено</button>
    <button data-outcome="false_positive">Ложное срабатывание</button>
    <button data-outcome="accepted_risk">Принятый риск</button>
  </div>

  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Write the stylesheet tokens**

В начало `web/style.css` — палитра на переменных, обе темы. Остальные правила
пишутся поверх этих токенов и ни одного цвета не задают напрямую.

```css
:root {
  --bg: #f6f7f9;
  --panel: #ffffff;
  --line: #d8dce3;
  --ink: #1b1f27;
  --muted: #5c6472;
  --accent: #1c5fd6;
  --warn: #b3261e;
  --ok: #1a7f4b;
  --row-hover: #eef2f8;
  --mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
}

[data-theme="dark"] {
  --bg: #14171c;
  --panel: #1b1f26;
  --line: #2d333d;
  --ink: #e6e9ef;
  --muted: #9aa3b2;
  --accent: #6da2ff;
  --warn: #ff8a80;
  --ok: #6ee7a8;
  --row-hover: #222833;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 13px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
```

Плотность: строка очереди высотой около 28 пикселей, идентификаторы моноширинным
шрифтом через `--mono`, отступы 6–8 пикселей. Никаких теней и скруглений больше
трёх пикселей — это рабочая панель, а не витрина.

- [ ] **Step 5: Write the script**

`web/app.js`. Состояние — один объект; каждое изменение фильтра перерисовывает
очередь.

```javascript
const state = {
  tab: "queue",
  filters: { q: "", state: "open", outcome: "", group: "none" },
  selected: new Set(),
  rows: [],
  groups: [],
};

const escape = (value) =>
  String(value).replace(/[&<>"']/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);

function theme(next) {
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (error) { /* private window */ }
}

async function loadQueue() {
  const query = new URLSearchParams();
  const { q, state: view, outcome, group } = state.filters;
  if (q) query.set("q", q);
  query.set("state", view);
  if (outcome && view !== "open") query.set("outcome", outcome);
  query.set("group", group);

  const payload = await (await fetch(`/api/incidents?${query}`)).json();
  state.rows = payload.incidents;
  state.groups = payload.groups;
  renderCounts(payload);
  renderQueue();
}

async function decide(outcome) {
  if (!state.selected.size) return;
  await fetch("/api/decisions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      incidents: [...state.selected],
      outcome,
      note: document.querySelector("#note").value,
    }),
  });
  state.selected.clear();
  document.querySelector("#note").value = "";
  await loadQueue();
}
```

Функции `renderQueue`, `renderCounts`, `renderGroups`, `renderDetail` и
`renderHistory` строят разметку из `state`. Каждое место, где в разметку попадает
идентификатор узла, обязано проходить через `escape`: имена приходят из чужой
системы.

Тема восстанавливается при загрузке: `theme(localStorage.getItem("theme") || "light")`
в блоке `try`, потому что в приватном окне обращение к хранилищу бросает.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/service/ -v`
Expected: PASS.

- [ ] **Step 7: Look at it**

Run: `uv run rga serve --config configs/service/full.yaml --port 8001`
Expected: плотный список, работающие фильтры, переключатель темы.

- [ ] **Step 8: Commit**

```bash
uv run ruff check .
git add web/index.html web/style.css web/app.js tests/service/test_page.py
git commit -m "feat: rebuild the page as a dense two-theme console"
git push origin main
```

---

### Task 10: Страница — выбор, массовое решение, группы, история

**Files:**
- Modify: `web/app.js`, `web/style.css`
- Test: `tests/service/test_page.py`

- [ ] **Step 1: Write the failing test**

```python
def test_the_script_wires_selection_and_decisions(client) -> None:
    """The page is untestable in a browser here, so the wiring is checked as text."""
    script = client.get("/static/app.js").text

    for marker in ("/api/decisions", "state.selected", "data-outcome", "/api/decisions?limit"):
        assert marker in script
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/service/test_page.py -v -k wiring`
Expected: FAIL — истории и массового решения в скрипте ещё нет.

- [ ] **Step 3: Write the implementation**

Дописать в `web/app.js`:

- обработчик на чекбоксах, складывающий идентификаторы в `state.selected` и
  показывающий `#selection`, когда набор не пуст;
- `Shift`-клик выделяет диапазон от предыдущего отмеченного до текущего;
- в заголовке группы кнопка «разобрать группу» добавляет в `state.selected` все её
  идентификаторы из `group.incidents`;
- кнопки `[data-outcome]` вызывают `decide(button.dataset.outcome)`;
- `renderHistory` читает `/api/decisions?limit=200` и строит таблицу: время, исход,
  заметка, субъект, отношение, объект, оценка;
- переключение вкладок прячет `#queue` и `#detail`, показывает `#history`, и
  наоборот, через свойство `hidden`.

В `#detail` после карточки добавить блок текущего решения и список истории по
инциденту — данные приходят в полях `decision` и `history` ручки
`/api/incidents/{id}`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -m "not integration and not gpu" -q`
Expected: PASS.

- [ ] **Step 5: Walk the whole flow by hand**

Поднять сервис, выбрать три изменения одного субъекта через группировку, поставить
«принятый риск» с заметкой, убедиться, что они ушли из очереди, найти их во вкладке
«История», открыть одно из них через `state=resolved` и переоткрыть.

- [ ] **Step 6: Commit**

```bash
uv run ruff check .
git add web/app.js web/style.css tests/service/test_page.py
git commit -m "feat: decide on several changes at once and keep the history"
git push origin main
```

---

### Task 11: Закрытие модуля

**Files:**
- Create: `docs/module-6-findings.md`
- Modify: `docs/findings.md`, `docs/demo.md`, `README.md`, `CLAUDE.md`

- [ ] **Step 1: Write the findings**

`docs/module-6-findings.md` по образцу предыдущих: что построено, какие решения
приняты и почему, что намеренно не сделано. Обязательно назвать:

- почему журнал только на дозапись, а не изменяемый флаг;
- почему запись решения самодостаточна (окно движется);
- **почему отказались от правил подавления** — система не прячет изменения молча;
- **почему решения не попадают в модель**, и что это проверяется тестом, а не
  обещанием;
- почему рисунок показывает верхушку по вкладу, а не окрестность.

- [ ] **Step 2: Update the index**

В `docs/findings.md` добавить раздел об инструменте триажа в карту документов и
одно-два утверждения в раздел 3 «о системе и объяснимости».

- [ ] **Step 3: Update the demonstration order**

В `docs/demo.md` добавить шаг: разобрать группу изменений одного субъекта и
показать историю. Это и есть ответ на вопрос «а что специалист с этим делает».

- [ ] **Step 4: Update README and CLAUDE.md**

README — две-три строки на английском про журнал решений. `CLAUDE.md` — закрыть
модуль 6 в разделе «Текущее состояние» и добавить документы в таблицу.

- [ ] **Step 5: Final verification**

```bash
uv run pytest -m "not integration and not gpu" -q
uv run ruff check .
git status --short
```

- [ ] **Step 6: Commit**

```bash
git add docs/module-6-findings.md docs/findings.md docs/demo.md README.md
git commit -m "docs: close module 6"
git push origin main
```
