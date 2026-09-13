"""Incremental graph state, queried as features are extracted.

Feature extraction walks the journal once. Before each grant is applied, the
features of that grant are read off this structure, so they describe the graph as
it stood immediately before the edge appeared. That is what makes the temporal
split honest: nothing a candidate can see happened after it.

The alternative — replaying the graph up to each candidate's timestamp — is
quadratic, and computing every candidate against one fixed snapshot leaks the
future into the past for anything inside the training span.
"""

from __future__ import annotations

from collections import deque
from math import log

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.util.timeutil import DAY_MS

#: Activity older than this is dropped, bounding memory. The widest activity
#: window a feature asks for must not exceed it.
ACTIVITY_HORIZON_MS = 7 * DAY_MS


class FeatureContext:
    """Mutable view of the graph at one point in time."""

    def __init__(self) -> None:
        self._out: dict[int, dict[str, set[str]]] = {int(r): {} for r in RelationType}
        self._in: dict[int, dict[str, set[str]]] = {int(r): {} for r in RelationType}
        self._undirected: dict[str, set[str]] = {}
        #: How many relations currently connect an unordered pair.
        self._pair_count: dict[tuple[str, str], int] = {}
        #: Edge existence and level, keyed the same way as GraphEvent.edge_key().
        self._edges: dict[tuple[str, int, str], int] = {}
        self._created: dict[str, int] = {}
        self._last_change: dict[str, int] = {}
        self._activity: dict[str, deque[int]] = {}

    # ── mutation ────────────────────────────────────────────────────────────

    def apply(self, event: GraphEvent) -> None:
        """Advance the context by one journal event."""
        self._touch(event.subject, event.ts)
        self._touch(event.object, event.ts)
        if event.actor is not None:
            self._touch(event.actor, event.ts)
            self._last_change[event.actor] = event.ts
            self._record_activity(event.actor, event.ts)

        self._last_change[event.subject] = event.ts
        self._record_activity(event.subject, event.ts)

        key = event.edge_key()
        if event.op is EventOp.GRANT:
            if key not in self._edges:
                self._link(event.subject, event.relation, event.object)
            self._edges[key] = int(event.level)
        elif key in self._edges:
            del self._edges[key]
            self._unlink(event.subject, event.relation, event.object)

    def _touch(self, node: str, ts: int) -> None:
        self._created.setdefault(node, ts)
        self._undirected.setdefault(node, set())

    def _record_activity(self, node: str, ts: int) -> None:
        history = self._activity.setdefault(node, deque())
        history.append(ts)
        horizon = ts - ACTIVITY_HORIZON_MS
        while history and history[0] < horizon:
            history.popleft()

    def _link(self, subject: str, relation: RelationType, target: str) -> None:
        self._out[int(relation)].setdefault(subject, set()).add(target)
        self._in[int(relation)].setdefault(target, set()).add(subject)
        if subject == target:
            return
        pair = (subject, target) if subject < target else (target, subject)
        count = self._pair_count.get(pair, 0)
        self._pair_count[pair] = count + 1
        if count == 0:
            self._undirected[subject].add(target)
            self._undirected[target].add(subject)

    def _unlink(self, subject: str, relation: RelationType, target: str) -> None:
        self._out[int(relation)].get(subject, set()).discard(target)
        self._in[int(relation)].get(target, set()).discard(subject)
        if subject == target:
            return
        pair = (subject, target) if subject < target else (target, subject)
        count = self._pair_count.get(pair, 0) - 1
        if count <= 0:
            self._pair_count.pop(pair, None)
            self._undirected[subject].discard(target)
            self._undirected[target].discard(subject)
        else:
            self._pair_count[pair] = count

    # ── queries ─────────────────────────────────────────────────────────────

    def knows(self, node: str) -> bool:
        return node in self._created

    def degree(self, node: str, relation: RelationType, *, incoming: bool = False) -> int:
        table = self._in if incoming else self._out
        return len(table[int(relation)].get(node, ()))

    def neighbours(self, node: str) -> set[str]:
        return self._undirected.get(node, set())

    def common_neighbours(self, a: str, b: str) -> int:
        return len(self.neighbours(a) & self.neighbours(b))

    def jaccard(self, a: str, b: str) -> float:
        left, right = self.neighbours(a), self.neighbours(b)
        union = len(left | right)
        return len(left & right) / union if union else 0.0

    def adamic_adar(self, a: str, b: str) -> float:
        """Shared neighbours weighted down by how many links they already have.

        A hub everyone touches says little; a neighbour shared by only these two
        says a lot.
        """
        total = 0.0
        for shared in self.neighbours(a) & self.neighbours(b):
            degree = len(self.neighbours(shared))
            if degree > 1:
                total += 1.0 / log(degree)
        return total

    def path_length(self, a: str, b: str, *, cap: int = 3) -> int | None:
        """Undirected hop distance, or None if farther than `cap` or disconnected."""
        if not self.knows(a) or not self.knows(b):
            return None
        if a == b:
            return 0

        frontier = {a}
        seen = {a}
        for distance in range(1, cap + 1):
            nxt: set[str] = set()
            for node in frontier:
                for neighbour in self._undirected.get(node, ()):
                    if neighbour == b:
                        return distance
                    if neighbour not in seen:
                        seen.add(neighbour)
                        nxt.add(neighbour)
            if not nxt:
                return None
            frontier = nxt
        return None

    def level_on(self, subject: str, target: str) -> PermissionLevel:
        """Permission the subject holds directly on the target."""
        key = (subject, int(RelationType.HAS_PERMISSION), target)
        return PermissionLevel(self._edges.get(key, int(PermissionLevel.NONE)))

    def max_level_of(self, node: str) -> PermissionLevel:
        """Strongest permission the node holds anywhere."""
        best = int(PermissionLevel.NONE)
        for target in self._out[int(RelationType.HAS_PERMISSION)].get(node, ()):
            key = (node, int(RelationType.HAS_PERMISSION), target)
            best = max(best, self._edges.get(key, 0))
        return PermissionLevel(best)

    def group_count(self, node: str) -> int:
        return self.degree(node, RelationType.MEMBER_OF)

    def created(self, node: str) -> int | None:
        return self._created.get(node)

    def last_change(self, node: str) -> int | None:
        return self._last_change.get(node)

    def activity(self, node: str, window_ms: int, now: int) -> int:
        """How many changes this node initiated within the trailing window."""
        history = self._activity.get(node)
        if not history:
            return 0
        threshold = now - window_ms
        return sum(1 for ts in history if ts >= threshold)
