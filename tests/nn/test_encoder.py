"""The stack of layers that turns node inputs into representations."""

import torch

from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.config import ModelConfig
from rga.nn.encoder import GraphEncoder
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
    builder.add_edge("bucket:logs", RelationType.PARENT_OF, "object:logs/a", created=30)
    return builder.build()


def test_the_encoder_returns_one_representation_per_node() -> None:
    torch.manual_seed(5)
    graph = _graph()
    tensors = graph_tensors(graph, device=CPU)
    config = ModelConfig(hidden_dim=16, num_layers=2, dropout=0.0)
    encoder = GraphEncoder(NODE_INPUT_DIM, config).eval()

    out = encoder(node_input_features(tensors), tensors)

    assert out.shape == (graph.num_nodes, 16)
    assert torch.isfinite(out).all()


def test_three_layers_carry_information_the_length_of_the_authorization_path() -> None:
    torch.manual_seed(5)
    graph = _graph()
    tensors = graph_tensors(graph, device=CPU)
    encoder = GraphEncoder(
        NODE_INPUT_DIM, ModelConfig(hidden_dim=16, num_layers=3, dropout=0.0)
    ).eval()
    inputs = node_input_features(tensors)

    baseline = encoder(inputs, tensors)
    moved = inputs.clone()
    moved[graph.index_of("user:alice")] += 1.0
    changed = encoder(moved, tensors)

    far = graph.index_of("object:logs/a")
    assert not torch.allclose(baseline[far], changed[far], atol=1e-6)
