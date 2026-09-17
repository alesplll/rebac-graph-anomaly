"""Every node's starting vector, derived from structure alone."""

import torch

from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.graph_tensors import graph_tensors
from rga.nn.node_inputs import NODE_INPUT_DIM, node_input_features

CPU = torch.device("cpu")


def _graph():
    builder = GraphBuilder()
    builder.add_edge("user:alice", RelationType.MEMBER_OF, "group:devops", created=10)
    builder.add_edge(
        "group:devops",
        RelationType.HAS_PERMISSION,
        "bucket:logs",
        level=int(PermissionLevel.ADMIN),
        created=20,
    )
    builder.register_node("user:lonely", 30)
    return builder.build()


def test_one_row_per_node_of_the_declared_width() -> None:
    graph = _graph()

    features = node_input_features(graph_tensors(graph, device=CPU))

    assert features.shape == (graph.num_nodes, NODE_INPUT_DIM)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()


def test_an_isolated_node_carries_only_its_kind() -> None:
    graph = _graph()

    features = node_input_features(graph_tensors(graph, device=CPU))
    row = features[graph.index_of("user:lonely")]

    assert row.sum().item() == 1.0


def test_a_node_holding_admin_records_the_strongest_level_it_holds() -> None:
    graph = _graph()

    features = node_input_features(graph_tensors(graph, device=CPU))
    devops = features[graph.index_of("group:devops")]
    logs = features[graph.index_of("bucket:logs")]

    assert devops[-2].item() == float(PermissionLevel.ADMIN)
    assert logs[-1].item() == float(PermissionLevel.ADMIN)
