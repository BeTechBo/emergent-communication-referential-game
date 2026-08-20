"""
tests/test_metrics.py
-----------------------------------------------------------------------------
Tests for Phase 5: Metric computation (Entropy, Topographic Similarity).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import message_entropy, topographic_similarity, levenshtein_distance


class MockSender(torch.nn.Module):
    """
    A mock Sender that outputs deterministic messages based on a provided
    mapping, for testing metrics.
    """
    def __init__(self, mapping: torch.Tensor):
        super().__init__()
        # mapping: (N, L, vocab_size)
        self.mapping = mapping

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        # Just return the stored logits directly, ignoring concepts
        # (Assuming the test passes the exact N concepts in the exact order)
        return self.mapping


def test_entropy_identical_messages():
    """If all messages are identical, entropy should be 0."""
    N, L, V = 10, 3, 5
    # Logits: symbol 1 is always argmax
    logits = torch.zeros(N, L, V)
    logits[:, :, 1] = 10.0
    
    sender = MockSender(logits)
    dummy_concepts = torch.zeros(N, 1)
    
    ent = message_entropy(sender, dummy_concepts)
    assert np.isclose(ent, 0.0), f"Expected 0.0 entropy, got {ent}"


def test_entropy_uniform_messages():
    """If all N messages are unique, entropy should be log2(N)."""
    N, L, V = 8, 3, 5
    logits = torch.zeros(N, L, V)
    
    # Create 8 unique messages by varying the first symbol
    for i in range(N):
        logits[i, 0, i % V] = 10.0
        logits[i, 1, (i // V) % V] = 10.0
        logits[i, 2, 0] = 10.0
        
    sender = MockSender(logits)
    dummy_concepts = torch.zeros(N, 1)
    
    ent = message_entropy(sender, dummy_concepts)
    expected = np.log2(N)
    assert np.isclose(ent, expected), f"Expected {expected}, got {ent}"


def test_levenshtein_distance():
    assert levenshtein_distance([1, 2, 3], [1, 2, 3]) == 0
    assert levenshtein_distance([1, 2, 3], [1, 2, 4]) == 1
    assert levenshtein_distance([1, 2, 3], [1, 3]) == 1  # delete 2
    assert levenshtein_distance([1, 2], [1, 3, 2]) == 1  # insert 3


def test_topographic_similarity_compositional():
    """
    A perfectly compositional mapping should yield topsim close to 1.0.
    """
    N = 4
    # Concepts: 2 attributes, vocab size 2 each. Total concept_dim = 4
    # Let's make 4 concepts: (0,0), (0,1), (1,0), (1,1)
    # One-hot representations:
    concepts = torch.tensor([
        [1, 0, 1, 0], # (0,0)
        [1, 0, 0, 1], # (0,1)
        [0, 1, 1, 0], # (1,0)
        [0, 1, 0, 1], # (1,1)
    ], dtype=torch.float32)
    
    # Perfectly compositional messages: Length 2.
    # Pos 0 = Attr 0 (values 1 or 2)
    # Pos 1 = Attr 1 (values 1 or 2)
    L, V = 2, 5
    logits = torch.zeros(N, L, V)
    
    # (0,0) -> [1, 1]
    logits[0, 0, 1] = 10.0; logits[0, 1, 1] = 10.0
    # (0,1) -> [1, 2]
    logits[1, 0, 1] = 10.0; logits[1, 1, 2] = 10.0
    # (1,0) -> [2, 1]
    logits[2, 0, 2] = 10.0; logits[2, 1, 1] = 10.0
    # (1,1) -> [2, 2]
    logits[3, 0, 2] = 10.0; logits[3, 1, 2] = 10.0
    
    sender = MockSender(logits)
    
    topsim = topographic_similarity(sender, concepts)
    assert np.isclose(topsim, 1.0), f"Expected topsim 1.0, got {topsim}"


def test_topographic_similarity_random():
    """
    A random mapping should yield topsim close to 0.0.
    """
    # Use more concepts to get a stable near-zero correlation
    N = 20
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Random one-hot concepts (2 attrs, vocab 5 each)
    concepts = torch.zeros(N, 10)
    for i in range(N):
        concepts[i, np.random.randint(0, 5)] = 1.0
        concepts[i, 5 + np.random.randint(0, 5)] = 1.0
        
    L, V = 4, 10
    # Random messages
    logits = torch.randn(N, L, V)
    
    sender = MockSender(logits)
    
    topsim = topographic_similarity(sender, concepts)
    # With N=20, it won't be EXACTLY 0, but should be small (e.g. within [-0.4, 0.4])
    assert abs(topsim) < 0.4, f"Expected topsim near 0, got {topsim}"
