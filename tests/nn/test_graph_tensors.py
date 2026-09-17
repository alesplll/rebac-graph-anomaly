"""The graph as index tensors — structure only."""

from dataclasses import replace

import numpy as np
import torch

from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.graph_tensors import NUM_SLOTS, SLOTS, graph_tensors

CPU = torch.device("cpu")


def _graph():
    """alice -> devops (member), devops -> logs (admin), logs -> logs/app (parent)."""
    builder = GraphBuilder()
    builder.add_edge("user:alice", RelationType.MEMBER_OF, "group:devops", created=10)
    builder.add_edge(
        "group:devops",
        RelationType.HAS_PERMISSION,
        "bucket:logs",
        level=int(PermissionLevel.ADMIN),
        created=20,
        actor="user:root",
    )
    builder.add_edge("bucket:logs", RelationType.PARENT_OF, "object:logs/app.log", created=30)
    return builder.build()


def test_there_are_eight_slots() -> None:
    assert NUM_SLOTS == 8
    assert len(set(SLOTS)) == 8


def test_messages_run_from_source_to_target_on_the_outgoing_slot() -> None:
    graph = _graph()
    tensors = graph_tensors(graph, device=CPU)
    slot = SLOTS.index((RelationType.MEMBER_OF, False))

    assert tensors.src[slot].tolist() == [graph.index_of("user:alice")]
    assert tensors.dst[slot].tolist() == [graph.index_of("group:devops")]


def test_the_incoming_slot_reverses_the_same_edge() -> None:
    graph = _graph()
    tensors = graph_tensors(graph, device=CPU)
    slot = SLOTS.index((RelationType.MEMBER_OF, True))

    assert tensors.src[slot].tolist() == [graph.index_of("group:devops")]
    assert tensors.dst[slot].tolist() == [graph.index_of("user:alice")]


def test_only_permission_edges_carry_a_level() -> None:
    tensors = graph_tensors(_graph(), device=CPU)

    for slot, (relation, _) in enumerate(SLOTS):
        levels = set(tensors.level[slot].tolist())
        if relation is RelationType.HAS_PERMISSION:
            assert levels == {int(PermissionLevel.ADMIN)}
        else:
            assert levels <= {int(PermissionLevel.NONE)}


def test_the_normalisation_is_one_over_root_degree() -> None:
    tensors = graph_tensors(_graph(), device=CPU)
    slot = SLOTS.index((RelationType.MEMBER_OF, False))
    devops = _graph().index_of("group:devops")

    assert tensors.norm[slot][devops].item() == 1.0


def test_times_and_actors_do_not_reach_the_tensors() -> None:
    graph = _graph()
    wiped = replace(
        graph,
        edge_created=np.full_like(graph.edge_created, -1),
        edge_actor=np.full_like(graph.edge_actor, -1),
        node_created=np.full_like(graph.node_created, -1),
    )

    original = graph_tensors(graph, device=CPU)
    blind = graph_tensors(wiped, device=CPU)

    for slot in range(NUM_SLOTS):
        assert torch.equal(original.src[slot], blind.src[slot])
        assert torch.equal(original.dst[slot], blind.dst[slot])
        assert torch.equal(original.level[slot], blind.level[slot])
        assert torch.equal(original.norm[slot], blind.norm[slot])


def test_keeping_a_subset_drops_the_rest() -> None:
    graph = _graph()
    keep = np.array([True, False, True])

    tensors = graph_tensors(graph, device=CPU, keep=keep)

    total = sum(int(tensors.src[slot].numel()) for slot in range(NUM_SLOTS))
    assert total == 4  # two edges, two directions each
