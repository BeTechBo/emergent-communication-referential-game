"""
data/synthetic_generator.py
─────────────────────────────────────────────────────────────────────────────
Phase 1 – Synthetic data generator for the emergent-communication referential
game (Havrylov & Titov, NeurIPS 2017).

Concepts are tuples of discrete attribute values, e.g. (shape, color, size).
Each concept is encoded as a fixed-size float32 tensor formed by concatenating
one-hot sub-vectors, one per attribute.  All hyper-parameters are configurable;
nothing is hard-coded.

Public API
──────────
ConceptSpace(vocab_sizes)
    Represents the full set of possible concepts given per-attribute vocab sizes.

encode_concept(concept, vocab_sizes) -> torch.Tensor
    Converts a raw concept tuple to a one-hot-concatenated float32 vector.

sample_batch(batch_size, *, vocab_sizes, n_distractors, rng) -> dict
    Returns a dict with keys:
      "target_vectors"     – (B, D) float32 tensor
      "distractor_vectors" – (B, N, D) float32 tensor
      "target_indices"     – (B,) int64 tensor  (position of target among all
                              N+1 candidates, randomly shuffled per sample)
      "all_candidates"     – (B, N+1, D) float32 tensor  (target + distractors,
                              in the shuffled order that target_indices indexes)
"""

from __future__ import annotations

import itertools
import random
from typing import List, Optional, Sequence, Tuple

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Default concept-space parameters
# ─────────────────────────────────────────────────────────────────────────────

# Three attributes with vocab sizes chosen so the total concept count is
# 4 × 4 × 5 = 80 distinct concepts — a comfortable default for fast
# experimentation without being trivially small.
DEFAULT_VOCAB_SIZES: Tuple[int, ...] = (4, 4, 5)   # 80 concepts
DEFAULT_N_DISTRACTORS: int = 4


# ─────────────────────────────────────────────────────────────────────────────
# ConceptSpace helper
# ─────────────────────────────────────────────────────────────────────────────

class ConceptSpace:
    """
    Enumerates all possible concepts for a given set of attribute vocab sizes.

    Parameters
    ----------
    vocab_sizes : sequence of int
        Number of distinct values for each attribute.  E.g. ``(4, 4, 5)``
        gives 80 concepts across three attributes.
    """

    def __init__(self, vocab_sizes: Sequence[int]) -> None:
        if len(vocab_sizes) == 0:
            raise ValueError("`vocab_sizes` must have at least one attribute.")
        if any(v < 2 for v in vocab_sizes):
            raise ValueError("Every attribute must have at least 2 values.")

        self.vocab_sizes: Tuple[int, ...] = tuple(int(v) for v in vocab_sizes)
        self.n_attributes: int = len(self.vocab_sizes)
        self.vector_dim: int = sum(self.vocab_sizes)  # length of one-hot concat

        # All concepts as a list of tuples, e.g. (0, 2, 3)
        self._all_concepts: List[Tuple[int, ...]] = list(
            itertools.product(*[range(v) for v in self.vocab_sizes])
        )
        self.n_concepts: int = len(self._all_concepts)

    def __len__(self) -> int:
        return self.n_concepts

    def __repr__(self) -> str:
        return (
            f"ConceptSpace(vocab_sizes={self.vocab_sizes}, "
            f"n_concepts={self.n_concepts}, vector_dim={self.vector_dim})"
        )

    @property
    def all_concepts(self) -> List[Tuple[int, ...]]:
        """Read-only view of every concept tuple."""
        return list(self._all_concepts)


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_concept(
    concept: Tuple[int, ...],
    vocab_sizes: Sequence[int],
) -> torch.Tensor:
    """
    Encode a single concept tuple as a concatenated one-hot float32 vector.

    Parameters
    ----------
    concept : tuple of int
        Attribute values, e.g. ``(2, 0, 3)``.
    vocab_sizes : sequence of int
        Number of distinct values per attribute.  Must match len(concept).

    Returns
    -------
    torch.Tensor, shape ``(sum(vocab_sizes),)``, dtype ``float32``.
    """
    if len(concept) != len(vocab_sizes):
        raise ValueError(
            f"concept length {len(concept)} does not match "
            f"vocab_sizes length {len(vocab_sizes)}."
        )
    parts: List[torch.Tensor] = []
    for attr_val, vocab_size in zip(concept, vocab_sizes):
        one_hot = torch.zeros(vocab_size, dtype=torch.float32)
        one_hot[attr_val] = 1.0
        parts.append(one_hot)
    return torch.cat(parts)  # shape: (sum(vocab_sizes),)


def encode_concept_batch(
    concepts: List[Tuple[int, ...]],
    vocab_sizes: Sequence[int],
) -> torch.Tensor:
    """
    Encode a list of concept tuples into a 2-D float32 tensor.

    Returns
    -------
    torch.Tensor, shape ``(len(concepts), sum(vocab_sizes))``, dtype ``float32``.
    """
    return torch.stack([encode_concept(c, vocab_sizes) for c in concepts])


# ─────────────────────────────────────────────────────────────────────────────
# Batch sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_batch(
    batch_size: int,
    *,
    vocab_sizes: Sequence[int] = DEFAULT_VOCAB_SIZES,
    n_distractors: int = DEFAULT_N_DISTRACTORS,
    rng: Optional[random.Random] = None,
    concept_space: Optional[ConceptSpace] = None,
) -> dict:
    """
    Sample a batch of referential-game examples.

    Each example consists of:
    - one **target** concept,
    - ``n_distractors`` distinct distractor concepts (none equal to the target),
    - a randomly chosen **target_index** indicating where the target sits among
      the ``n_distractors + 1`` candidates (so the Receiver cannot learn a
      positional shortcut).

    Parameters
    ----------
    batch_size : int
        Number of examples to generate.
    vocab_sizes : sequence of int, optional
        Per-attribute vocabulary sizes (default ``(4, 4, 5)``).
    n_distractors : int, optional
        Number of distractor concepts per example (default 4).
    rng : random.Random, optional
        A seeded ``random.Random`` instance for full reproducibility.
        If ``None``, Python's module-level random state is used.
    concept_space : ConceptSpace, optional
        Pre-built concept space.  Constructed from ``vocab_sizes`` if not given.

    Returns
    -------
    dict with keys:
        ``"target_vectors"``      – ``(B, D)`` float32
        ``"distractor_vectors"``  – ``(B, N, D)`` float32
        ``"target_indices"``      – ``(B,)`` int64  (position of target in
                                    ``all_candidates``)
        ``"all_candidates"``      – ``(B, N+1, D)`` float32

    where ``B = batch_size``, ``N = n_distractors``,
    ``D = sum(vocab_sizes)``.
    """
    if rng is None:
        rng = random.Random()  # unseeded – non-reproducible by default

    if concept_space is None:
        concept_space = ConceptSpace(vocab_sizes)

    if n_distractors < 1:
        raise ValueError("`n_distractors` must be >= 1.")
    if n_distractors >= concept_space.n_concepts:
        raise ValueError(
            f"`n_distractors` ({n_distractors}) must be less than the total "
            f"number of concepts ({concept_space.n_concepts})."
        )

    all_concepts = concept_space.all_concepts  # list copy, safe to sample from
    D = concept_space.vector_dim
    N = n_distractors

    target_vecs_list: List[torch.Tensor] = []
    distractor_vecs_list: List[torch.Tensor] = []
    target_indices_list: List[int] = []
    all_candidates_list: List[torch.Tensor] = []

    for _ in range(batch_size):
        # 1. Sample target
        target: Tuple[int, ...] = rng.choice(all_concepts)

        # 2. Sample N distinct distractors, none equal to the target
        pool = [c for c in all_concepts if c != target]
        distractors: List[Tuple[int, ...]] = rng.sample(pool, N)

        # 3. Encode all
        target_vec = encode_concept(target, concept_space.vocab_sizes)
        distractor_vecs = [
            encode_concept(d, concept_space.vocab_sizes) for d in distractors
        ]

        # 4. Build candidate list with target inserted at a random position
        target_idx: int = rng.randrange(N + 1)
        candidates: List[torch.Tensor] = list(distractor_vecs)
        candidates.insert(target_idx, target_vec)  # N+1 total
        candidates_tensor = torch.stack(candidates)  # (N+1, D)

        target_vecs_list.append(target_vec)
        distractor_vecs_list.append(torch.stack(distractor_vecs))  # (N, D)
        target_indices_list.append(target_idx)
        all_candidates_list.append(candidates_tensor)

    return {
        # (B, D)
        "target_vectors": torch.stack(target_vecs_list),
        # (B, N, D)
        "distractor_vectors": torch.stack(distractor_vecs_list),
        # (B,)
        "target_indices": torch.tensor(target_indices_list, dtype=torch.long),
        # (B, N+1, D)
        "all_candidates": torch.stack(all_candidates_list),
    }
