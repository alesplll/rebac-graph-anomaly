"""Analytic gradients of the hand-written layer against finite differences."""

import torch

from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.graph_tensors import graph_tensors
from rga.nn.layers import RelationalLayer

CPU = torch.device("cpu")


def _small_graph():
    """Small enough for finite differences, wide enough to use every slot."""
    builder = GraphBuilder()
    builder.add_edge("user:a", RelationType.MEMBER_OF, "group:g", created=1)
    builder.add_edge("user:b", RelationType.MEMBER_OF, "group:g", created=2)
    builder.add_edge(
        "group:g",
        RelationType.HAS_PERMISSION,
        "bucket:x",
        level=int(PermissionLevel.ADMIN),
        created=3,
    )
    builder.add_edge("bucket:x", RelationType.PARENT_OF, "object:x/k", created=4)
    builder.add_edge("user:a", RelationType.OWNER_OF, "bucket:x", created=5)
    return builder.build()


def test_layer_gradients_match_finite_differences() -> None:
    torch.manual_seed(0)
    graph = _small_graph()
    tensors = graph_tensors(graph, device=CPU)
    layer = RelationalLayer(3, 3, num_bases=2, dropout=0.0).double().eval()

    names = [name for name, _ in layer.named_parameters()]
    values = tuple(
        parameter.detach().clone().requires_grad_(True)
        for _, parameter in layer.named_parameters()
    )
    h = torch.randn(graph.num_nodes, 3, dtype=torch.double, requires_grad=True)

    def run(node_state, *parameters):
        bound = dict(zip(names, parameters, strict=True))
        return torch.func.functional_call(layer, bound, (node_state, tensors))

    assert torch.autograd.gradcheck(run, (h, *values), eps=1e-6, atol=1e-4, rtol=1e-3)
