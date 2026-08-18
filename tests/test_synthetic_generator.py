"""
tests/test_synthetic_generator.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for data/synthetic_generator.py (Phase 1).

Test coverage
─────────────
1. Output shapes and dtypes are correct for the default configuration.
2. No distractor set ever contains a concept equal to the target.
3. Running sample_batch twice with the same seed produces identical output
   (reproducibility via the rng argument).
4. Custom vocab_sizes and n_distractors are correctly reflected in shapes.
5. ConceptSpace reports the right concept count and vector dimensionality.
6. encode_concept produces valid one-hot sub-vectors (each sums to 1, binary).
7. all_candidates contains the target at the index given by target_indices.
8. Guard-rail: requesting too many distractors raises ValueError.
"""

from __future__ import annotations

import random
import sys
import os

import pytest
import torch

# Make sure the repo root is on sys.path so "data.synthetic_generator" resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.synthetic_generator import (
    DEFAULT_N_DISTRACTORS,
    DEFAULT_VOCAB_SIZES,
    ConceptSpace,
    encode_concept,
    encode_concept_batch,
    sample_batch,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE = 64
SEED = 42


def make_rng(seed: int = SEED) -> random.Random:
    return random.Random(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ConceptSpace
# ─────────────────────────────────────────────────────────────────────────────

class TestConceptSpace:
    def test_default_n_concepts(self):
        """4*4*5 = 80 distinct concepts."""
        cs = ConceptSpace(DEFAULT_VOCAB_SIZES)
        expected = 1
        for v in DEFAULT_VOCAB_SIZES:
            expected *= v
        assert cs.n_concepts == expected

    def test_vector_dim(self):
        """vector_dim == sum of vocab sizes."""
        cs = ConceptSpace(DEFAULT_VOCAB_SIZES)
        assert cs.vector_dim == sum(DEFAULT_VOCAB_SIZES)

    def test_all_concepts_unique(self):
        """Every enumerated concept is unique."""
        cs = ConceptSpace(DEFAULT_VOCAB_SIZES)
        assert len(set(cs.all_concepts)) == cs.n_concepts

    def test_custom_vocab_sizes(self):
        vocab_sizes = (3, 5, 4, 2)
        cs = ConceptSpace(vocab_sizes)
        assert cs.n_concepts == 3 * 5 * 4 * 2
        assert cs.vector_dim == sum(vocab_sizes)

    def test_invalid_single_value_attribute(self):
        with pytest.raises(ValueError):
            ConceptSpace((4, 1, 5))  # an attribute with only 1 value

    def test_empty_vocab_sizes_raises(self):
        with pytest.raises(ValueError):
            ConceptSpace(())


# ─────────────────────────────────────────────────────────────────────────────
# 2. encode_concept
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeConcept:
    def test_shape(self):
        vec = encode_concept((1, 0, 3), DEFAULT_VOCAB_SIZES)
        assert vec.shape == (sum(DEFAULT_VOCAB_SIZES),)

    def test_dtype(self):
        vec = encode_concept((0, 0, 0), DEFAULT_VOCAB_SIZES)
        assert vec.dtype == torch.float32

    def test_valid_one_hot_per_attribute(self):
        """Each attribute block must sum to 1 and contain only 0/1 values."""
        vocab_sizes = (3, 4, 5)
        concept = (2, 1, 4)
        vec = encode_concept(concept, vocab_sizes)
        offset = 0
        for attr_val, vsize in zip(concept, vocab_sizes):
            block = vec[offset : offset + vsize]
            assert block.sum().item() == pytest.approx(1.0)
            assert set(block.tolist()).issubset({0.0, 1.0})
            assert block[attr_val].item() == 1.0
            offset += vsize

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            encode_concept((0, 1), (3, 4, 5))  # 2 values, 3 attr sizes

    def test_batch_encoding_shape(self):
        concepts = [(0, 0, 0), (1, 2, 3), (3, 3, 4)]
        batch = encode_concept_batch(concepts, DEFAULT_VOCAB_SIZES)
        assert batch.shape == (3, sum(DEFAULT_VOCAB_SIZES))
        assert batch.dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# 3. sample_batch — shapes and dtypes  (Requirement 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleBatchShapes:
    def test_target_vectors_shape(self):
        batch = sample_batch(BATCH_SIZE, rng=make_rng())
        D = sum(DEFAULT_VOCAB_SIZES)
        assert batch["target_vectors"].shape == (BATCH_SIZE, D)

    def test_distractor_vectors_shape(self):
        batch = sample_batch(BATCH_SIZE, rng=make_rng())
        D = sum(DEFAULT_VOCAB_SIZES)
        N = DEFAULT_N_DISTRACTORS
        assert batch["distractor_vectors"].shape == (BATCH_SIZE, N, D)

    def test_target_indices_shape(self):
        batch = sample_batch(BATCH_SIZE, rng=make_rng())
        assert batch["target_indices"].shape == (BATCH_SIZE,)

    def test_all_candidates_shape(self):
        batch = sample_batch(BATCH_SIZE, rng=make_rng())
        D = sum(DEFAULT_VOCAB_SIZES)
        N = DEFAULT_N_DISTRACTORS
        assert batch["all_candidates"].shape == (BATCH_SIZE, N + 1, D)

    def test_dtypes(self):
        batch = sample_batch(BATCH_SIZE, rng=make_rng())
        assert batch["target_vectors"].dtype == torch.float32
        assert batch["distractor_vectors"].dtype == torch.float32
        assert batch["all_candidates"].dtype == torch.float32
        assert batch["target_indices"].dtype == torch.long

    def test_custom_n_distractors(self):
        n = 7
        vocab_sizes = (5, 5, 5)
        batch = sample_batch(
            BATCH_SIZE,
            vocab_sizes=vocab_sizes,
            n_distractors=n,
            rng=make_rng(),
        )
        D = sum(vocab_sizes)
        assert batch["target_vectors"].shape == (BATCH_SIZE, D)
        assert batch["distractor_vectors"].shape == (BATCH_SIZE, n, D)
        assert batch["all_candidates"].shape == (BATCH_SIZE, n + 1, D)
        assert batch["target_indices"].shape == (BATCH_SIZE,)

    def test_custom_vocab_sizes(self):
        vocab_sizes = (3, 6, 4, 2)
        batch = sample_batch(
            BATCH_SIZE,
            vocab_sizes=vocab_sizes,
            rng=make_rng(),
        )
        D = sum(vocab_sizes)
        assert batch["target_vectors"].shape == (BATCH_SIZE, D)


# ─────────────────────────────────────────────────────────────────────────────
# 4. No distractor equals the target  (Requirement 2)
# ─────────────────────────────────────────────────────────────────────────────

class TestNoDistractorEqualsTarget:
    def test_large_batch_no_duplicates(self):
        """
        For every sample in a large batch, none of the distractor vectors
        should be equal to the target vector.
        """
        batch = sample_batch(512, rng=make_rng())
        targets = batch["target_vectors"]          # (B, D)
        distractors = batch["distractor_vectors"]  # (B, N, D)

        for i in range(targets.shape[0]):
            t = targets[i]          # (D,)
            ds = distractors[i]     # (N, D)
            for j in range(ds.shape[0]):
                assert not torch.equal(t, ds[j]), (
                    f"Sample {i}: distractor {j} equals the target."
                )

    def test_no_duplicate_distractors_within_sample(self):
        """
        Within a single sample, all distractors must be distinct from each
        other (and from the target, already tested above).
        """
        batch = sample_batch(256, rng=make_rng())
        distractors = batch["distractor_vectors"]  # (B, N, D)
        N = distractors.shape[1]

        for i in range(distractors.shape[0]):
            ds = distractors[i]  # (N, D)
            for a in range(N):
                for b in range(a + 1, N):
                    assert not torch.equal(ds[a], ds[b]), (
                        f"Sample {i}: distractor {a} == distractor {b}."
                    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reproducibility  (Requirement 3)
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_output(self):
        """Two calls with the same seed must produce byte-identical tensors."""
        batch1 = sample_batch(BATCH_SIZE, rng=make_rng(SEED))
        batch2 = sample_batch(BATCH_SIZE, rng=make_rng(SEED))

        for key in ("target_vectors", "distractor_vectors",
                    "target_indices", "all_candidates"):
            assert torch.equal(batch1[key], batch2[key]), (
                f"Key '{key}' differs between two runs with the same seed."
            )

    def test_different_seeds_different_output(self):
        """Two calls with different seeds should (almost certainly) differ."""
        batch1 = sample_batch(BATCH_SIZE, rng=make_rng(SEED))
        batch2 = sample_batch(BATCH_SIZE, rng=make_rng(SEED + 1))
        # It is astronomically unlikely that all targets are the same.
        assert not torch.equal(batch1["target_vectors"], batch2["target_vectors"])


# ─────────────────────────────────────────────────────────────────────────────
# 6. Target is at the correct index in all_candidates
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetIndexConsistency:
    def test_target_at_reported_index(self):
        """
        all_candidates[i, target_indices[i]] must equal target_vectors[i]
        for every sample in the batch.
        """
        batch = sample_batch(BATCH_SIZE, rng=make_rng())
        targets = batch["target_vectors"]        # (B, D)
        candidates = batch["all_candidates"]     # (B, N+1, D)
        indices = batch["target_indices"]        # (B,)

        for i in range(targets.shape[0]):
            idx = indices[i].item()
            assert torch.equal(targets[i], candidates[i, idx]), (
                f"Sample {i}: target not found at index {idx} in all_candidates."
            )

    def test_target_indices_range(self):
        """target_indices must be in [0, N]."""
        N = DEFAULT_N_DISTRACTORS
        batch = sample_batch(BATCH_SIZE, rng=make_rng())
        indices = batch["target_indices"]
        assert (indices >= 0).all() and (indices <= N).all(), (
            "Some target_indices are out of the valid range [0, N]."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Guard-rails
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardRails:
    def test_too_many_distractors_raises(self):
        """n_distractors >= n_concepts should raise ValueError."""
        with pytest.raises(ValueError):
            sample_batch(
                4,
                vocab_sizes=(2, 2),    # only 4 concepts
                n_distractors=4,       # need 4 distractors + 1 target = 5, but only 4 total
                rng=make_rng(),
            )

    def test_zero_distractors_raises(self):
        with pytest.raises(ValueError):
            sample_batch(4, n_distractors=0, rng=make_rng())
