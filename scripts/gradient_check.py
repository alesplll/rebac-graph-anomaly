"""Finite-difference check of the hand-written layer, as a table for the report.

Run: uv run python scripts/gradient_check.py
Writes: docs/thesis/gradcheck.md
"""

from __future__ import annotations

from pathlib import Path

import torch

from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.graph_tensors import graph_tensors
from rga.nn.layers import RelationalLayer

OUTPUT = Path("docs/thesis/gradcheck.md")
EPS = 1e-6


def _graph():
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


def _deviation(parameter, loss_of) -> float:
    """Largest absolute difference between the two gradients of one tensor."""
    analytic = parameter.grad.detach().clone()
    numeric = torch.zeros_like(parameter)
    flat = parameter.data.view(-1)

    for index in range(flat.numel()):
        original = flat[index].item()
        flat[index] = original + EPS
        plus = loss_of()
        flat[index] = original - EPS
        minus = loss_of()
        flat[index] = original
        numeric.view(-1)[index] = (plus - minus) / (2 * EPS)

    return float((analytic - numeric).abs().max())


def main() -> None:
    torch.manual_seed(0)
    graph = _graph()
    tensors = graph_tensors(graph, device=torch.device("cpu"))
    layer = RelationalLayer(3, 3, num_bases=2, dropout=0.0).double().eval()
    state = torch.randn(graph.num_nodes, 3, dtype=torch.double)

    def loss_of() -> float:
        with torch.no_grad():
            return float(layer(state, tensors).pow(2).sum())

    layer(state, tensors).pow(2).sum().backward()

    rows = [
        (name, tuple(parameter.shape), _deviation(parameter, loss_of))
        for name, parameter in layer.named_parameters()
    ]

    lines = [
        "### Проверка градиентов конечными разностями",
        "",
        "Аналитический градиент против центральной разности с шагом 1e-6, двойная",
        f"точность, граф из {graph.num_nodes} узлов и {graph.num_edges} рёбер.",
        "",
        "| параметр | форма | max &#124;аналитический − численный&#124; |",
        "|---|---|---|",
    ]
    for name, shape, deviation in rows:
        lines.append(f"| `{name}` | {tuple(shape)} | {deviation:.2e} |")
    lines.append("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT}")
    for name, _, deviation in rows:
        print(f"  {name}: {deviation:.2e}")


if __name__ == "__main__":
    main()
