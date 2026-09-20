"""The likelihood head, the reconstruction head and the correspondence head."""

import torch

from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.config import ModelConfig
from rga.nn.heads import CorrespondenceHead, EdgeLikelihoodHead, NodeReconstructionHead

CONFIG = ModelConfig(hidden_dim=8, embedding_dim=4, edge_hidden=8, dropout=0.0)


def test_the_likelihood_head_returns_one_logit_per_edge() -> None:
    torch.manual_seed(1)
    head = EdgeLikelihoodHead(node_dim=8, edge_dim=5, config=CONFIG).eval()

    logits = head(
        torch.randn(3, 8),
        torch.randn(3, 8),
        torch.tensor([int(RelationType.HAS_PERMISSION)] * 3),
        torch.tensor([int(PermissionLevel.ADMIN)] * 3),
        torch.randn(3, 5),
    )

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_the_likelihood_head_reacts_to_the_permission_level() -> None:
    torch.manual_seed(1)
    head = EdgeLikelihoodHead(node_dim=8, edge_dim=5, config=CONFIG).eval()
    source, target = torch.randn(1, 8), torch.randn(1, 8)
    relation = torch.tensor([int(RelationType.HAS_PERMISSION)])
    features = torch.randn(1, 5)

    read = head(source, target, relation, torch.tensor([int(PermissionLevel.READ)]), features)
    admin = head(source, target, relation, torch.tensor([int(PermissionLevel.ADMIN)]), features)

    assert not torch.allclose(read, admin, atol=1e-6)


def test_the_reconstruction_head_rebuilds_the_profile_shape() -> None:
    torch.manual_seed(1)
    head = NodeReconstructionHead(node_dim=8, output_dim=14, config=CONFIG).eval()

    rebuilt = head(torch.randn(6, 8))

    assert rebuilt.shape == (6, 14)


def test_deviation_is_zero_for_a_perfect_reconstruction() -> None:
    torch.manual_seed(1)
    head = NodeReconstructionHead(node_dim=8, output_dim=14, config=CONFIG).eval()
    state = torch.randn(6, 8)

    perfect = head.deviation(state, head(state))

    assert torch.allclose(perfect, torch.zeros(6), atol=1e-6)


def test_deviation_grows_with_the_error() -> None:
    torch.manual_seed(1)
    head = NodeReconstructionHead(node_dim=8, output_dim=14, config=CONFIG).eval()
    state = torch.randn(6, 8)
    target = head(state)

    near = head.deviation(state, target + 0.1)
    far = head.deviation(state, target + 1.0)

    assert (far > near).all()


def test_the_correspondence_head_returns_one_logit_per_pair() -> None:
    torch.manual_seed(1)
    head = CorrespondenceHead(node_dim=8, context_dim=5, config=CONFIG).eval()

    logits = head(torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 5))

    assert logits.shape == (4,)


def test_the_correspondence_head_reacts_to_the_context_row() -> None:
    """The whole point: a different row must move the logit.

    The likelihood head cannot learn this, because a positive and all of its
    corrupted variants carry the same row and the gradient never tells them apart.
    """
    torch.manual_seed(2)
    head = CorrespondenceHead(node_dim=8, context_dim=5, config=CONFIG).eval()
    h_src, h_dst = torch.randn(1, 8), torch.randn(1, 8)

    mine = head(h_src, h_dst, torch.zeros(1, 5))
    theirs = head(h_src, h_dst, torch.ones(1, 5))

    assert not torch.allclose(mine, theirs)
