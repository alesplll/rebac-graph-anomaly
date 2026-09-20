"""Every node's starting vector, derived from structure alone."""

from dataclasses import replace

import pytest
import torch

from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.config import ModelConfig
from rga.nn.graph_tensors import graph_tensors
from rga.nn.node_inputs import (
    NODE_INPUT_DIM,
    neighbourhood_targets,
    node_input_features,
    reconstruction_target,
)

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


def test_the_neighbourhood_target_averages_the_nodes_a_node_touches() -> None:
    """Rebuilding the surroundings rather than the node itself."""
    graph = _graph()
    tensors = graph_tensors(graph, device=CPU)
    inputs = node_input_features(tensors)

    targets = neighbourhood_targets(tensors, inputs)

    neighbours = [graph.index_of("user:alice"), graph.index_of("bucket:logs")]
    expected = inputs[neighbours].mean(dim=0)
    assert torch.allclose(targets[graph.index_of("group:devops")], expected, atol=1e-6)


def test_a_node_with_no_neighbours_reconstructs_itself() -> None:
    """There is nothing to average, and a row of zeros would make every isolate odd."""
    graph = _graph()
    tensors = graph_tensors(graph, device=CPU)
    inputs = node_input_features(tensors)

    targets = neighbourhood_targets(tensors, inputs)

    lonely = graph.index_of("user:lonely")
    assert torch.allclose(targets[lonely], inputs[lonely], atol=1e-6)


def test_the_configuration_decides_what_is_reconstructed() -> None:
    graph = _graph()
    tensors = graph_tensors(graph, device=CPU)
    inputs = node_input_features(tensors)

    profile = reconstruction_target(tensors, inputs, ModelConfig())
    neighbourhood = reconstruction_target(
        tensors, inputs, replace(ModelConfig(), reconstruction_target="neighbourhood")
    )

    assert torch.allclose(profile, inputs)
    assert not torch.allclose(neighbourhood, inputs)


def test_an_unknown_reconstruction_target_is_refused() -> None:
    graph = _graph()
    tensors = graph_tensors(graph, device=CPU)
    inputs = node_input_features(tensors)

    with pytest.raises(ValueError, match="reconstruction target"):
        reconstruction_target(
            tensors, inputs, replace(ModelConfig(), reconstruction_target="nonsense")
        )
