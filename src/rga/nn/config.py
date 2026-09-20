"""Hyperparameters of the network, in one frozen place.

Defaults are the spec's: hidden width 64, three layers — the radius that covers
user -> group -> bucket -> object, which is the path the authorization engine itself
walks. Experiment configs override what they need; nothing reads a loose float.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Everything the model and its training loop need to know."""

    hidden_dim: int = 64
    num_layers: int = 3
    num_bases: int = 4
    dropout: float = 0.1
    edge_hidden: int = 64
    embedding_dim: int = 16
    epochs: int = 200
    patience: int = 20
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    #: Corrupted edges drawn per real edge, spread across the sampling strategies.
    negatives_per_edge: int = 4
    #: Share of the training graph's edges held out to stop on.
    validation_share: float = 0.1
    #: Weight of the profile reconstruction term against the edge likelihood term.
    reconstruction_weight: float = 0.5
    #: What the reconstruction head rebuilds: "profile", the node's own attributes,
    #: or "neighbourhood", the mean attributes of the nodes it touches. The profile
    #: variant makes the deviation track node degree; see docs/module-3-findings.md.
    reconstruction_target: str = "profile"
    #: Weight of the correspondence term. Zero leaves that head out of the model.
    correspondence_weight: float = 0.0
