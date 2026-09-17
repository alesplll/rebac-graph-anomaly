"""One round of relational message passing."""

import torch

from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.graph_tensors import graph_tensors
from rga.nn.layers import RelationalLayer

CPU = torch.device("cpu")


def _graph(level: PermissionLevel = PermissionLevel.READ):
    builder = GraphBuilder()
    builder.add_edge("user:alice", RelationType.MEMBER_OF, "group:devops", created=10)
    builder.add_edge(
        "group:devops", RelationType.HAS_PERMISSION, "bucket:logs", level=int(level), created=20
    )
    builder.register_node("user:lonely", 30)
    return builder.build()


def _layer(dim: int = 5) -> RelationalLayer:
    torch.manual_seed(11)
    layer = RelationalLayer(dim, dim, num_bases=2, dropout=0.0)
    return layer.eval()


def test_output_has_one_row_per_node() -> None:
    graph = _graph()
    layer = _layer()
    h = torch.randn(graph.num_nodes, 5)

    out = layer(h, graph_tensors(graph, device=CPU))

    assert out.shape == (graph.num_nodes, 5)
    assert torch.isfinite(out).all()


def test_an_isolated_node_sees_only_its_own_self_loop() -> None:
    graph = _graph()
    layer = _layer()
    h = torch.randn(graph.num_nodes, 5)
    lonely = graph.index_of("user:lonely")

    out = layer(h, graph_tensors(graph, device=CPU))
    expected = layer.norm(h[lonely] + torch.nn.functional.gelu(layer.self_loop(h[lonely])))

    assert torch.allclose(out[lonely], expected, atol=1e-6)


def test_the_permission_level_changes_the_message() -> None:
    layer = _layer()
    h = torch.randn(_graph().num_nodes, 5)
    target = _graph().index_of("bucket:logs")

    low = layer(h, graph_tensors(_graph(PermissionLevel.READ), device=CPU))
    high = layer(h, graph_tensors(_graph(PermissionLevel.ADMIN), device=CPU))

    assert not torch.allclose(low[target], high[target], atol=1e-6)


def test_the_same_seed_gives_the_same_layer() -> None:
    graph = _graph()
    h = torch.randn(graph.num_nodes, 5)
    tensors = graph_tensors(graph, device=CPU)

    first = _layer()(h, tensors)
    second = _layer()(h, tensors)

    assert torch.equal(first, second)
