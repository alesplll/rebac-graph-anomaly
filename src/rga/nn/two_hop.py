"""How far apart the ends of a change stood before it happened.

Module 3 found that drawing hard negatives from the two-hop neighbourhood did not
help, and offered an explanation it could not check: a node two steps away is the
shape of a legitimate future grant, so teaching the model to reject those pairs
teaches it to hold down ordinary changes as well.

That explanation is measurable without training anything. If ordinary changes sit
at two hops far more often than anomalous ones do, the strategy is indeed spending
its negatives on the wrong shape.
"""

from __future__ import annotations

import numpy as np

from rga.domain.graph import AccessGraph
from rga.nn.negatives import undirected_csr

#: Further than two steps, unreachable, or with an endpoint the graph has not seen.
BEYOND = 3


def hop_distribution(graph: AccessGraph, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Undirected distance between the ends of every candidate, capped at `BEYOND`.

    The loop is Python on purpose: this runs once per study, not once per epoch, and
    a two-step search written out plainly is easier to trust than a clever one. The
    neighbour sets are built once so that each lookup inside it costs nothing.
    """
    indptr, indices = undirected_csr(graph)
    neighbours = [
        set(indices[indptr[node] : indptr[node + 1]].tolist())
        for node in range(graph.num_nodes)
    ]
    result = np.full(np.shape(src), BEYOND, dtype=np.int64)

    for position, (tail, head) in enumerate(zip(src, dst, strict=True)):
        if tail < 0 or head < 0:
            continue
        if tail == head:
            result[position] = 0
        elif head in neighbours[tail]:
            result[position] = 1
        elif any(head in neighbours[middle] for middle in neighbours[tail]):
            result[position] = 2

    return result
