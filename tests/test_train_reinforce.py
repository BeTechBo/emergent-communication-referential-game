"""
tests/test_train_reinforce.py
-----------------------------------------------------------------------------
Tests for Phase 4: REINFORCE training logic.
"""

from __future__ import annotations

import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.synthetic_generator import ConceptSpace, sample_batch
from src.sender import Sender
from src.receiver import Receiver
from src.train_reinforce import reinforce_generate, train_step, DEFAULT_CONFIG

# Small settings for fast tests
VOCAB_SIZES   = (2, 3)
CONCEPT_DIM   = sum(VOCAB_SIZES)   # 5
VOCAB_SIZE    = 5
EMBED_DIM     = 8
HIDDEN_DIM    = 16
MAX_LENGTH    = 3
BATCH_SIZE    = 4
N_DISTRACTORS = 2
BASELINE_INIT = 0.2
BASELINE_DECAY = 0.99


@pytest.fixture
def models():
    torch.manual_seed(42)
    sender   = Sender(CONCEPT_DIM, VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, MAX_LENGTH)
    receiver = Receiver(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, CONCEPT_DIM)
    return sender, receiver


@pytest.fixture
def batch():
    rng = random.Random(42)
    cs  = ConceptSpace(VOCAB_SIZES)
    return sample_batch(
        BATCH_SIZE,
        vocab_sizes=VOCAB_SIZES,
        n_distractors=N_DISTRACTORS,
        rng=rng,
        concept_space=cs,
    )


# ---------------------------------------------------------------------------
# Test 1: one training step produces finite losses
# ---------------------------------------------------------------------------

def test_train_step_finite_losses(models, batch):
    """One training step runs without error and produces finite losses."""
    sender, receiver = models
    optimizer = torch.optim.Adam(
        list(sender.parameters()) + list(receiver.parameters()), lr=0.01
    )
    snd_loss, rcv_loss, acc, new_baseline = train_step(
        sender, receiver, batch, optimizer, BASELINE_INIT, BASELINE_DECAY
    )

    assert isinstance(snd_loss,    float), "sender_loss should be a float"
    assert isinstance(rcv_loss,    float), "receiver_loss should be a float"
    assert isinstance(acc,         float), "accuracy should be a float"
    assert isinstance(new_baseline, float), "new_baseline should be a float"

    assert not torch.isnan(torch.tensor(snd_loss)), "sender_loss is NaN"
    assert not torch.isnan(torch.tensor(rcv_loss)), "receiver_loss is NaN"
    assert not torch.isinf(torch.tensor(snd_loss)), "sender_loss is Inf"
    assert not torch.isinf(torch.tensor(rcv_loss)), "receiver_loss is Inf"
    assert 0.0 <= acc <= 1.0, f"accuracy {acc} out of [0,1]"


# ---------------------------------------------------------------------------
# Test 2: gradients reach both Sender and Receiver after one step
# ---------------------------------------------------------------------------

def test_gradient_norms_sender_and_receiver(models, batch):
    """
    After one train_step, both Sender and Receiver must have non-zero gradients.
    This is the canonical way REINFORCE setups silently fail: forgetting to
    multiply log_probs by (reward - baseline), or wrong sign, leaves the
    Sender gradient at zero even when .grad is not None.
    """
    sender, receiver = models
    optimizer = torch.optim.Adam(
        list(sender.parameters()) + list(receiver.parameters()), lr=0.01
    )
    train_step(sender, receiver, batch, optimizer, BASELINE_INIT, BASELINE_DECAY)

    # Sender — check concept_encoder and lstm_cell
    sender_grad_norms = {}
    for name, param in sender.named_parameters():
        if param.grad is not None:
            sender_grad_norms[name] = param.grad.norm().item()

    # Receiver — check symbol_proj and concept_proj
    receiver_grad_norms = {}
    for name, param in receiver.named_parameters():
        if param.grad is not None:
            receiver_grad_norms[name] = param.grad.norm().item()

    print("\n--- Sender gradient norms ---")
    for name, norm in sender_grad_norms.items():
        print(f"  {name}: {norm:.6f}")

    print("--- Receiver gradient norms ---")
    for name, norm in receiver_grad_norms.items():
        print(f"  {name}: {norm:.6f}")

    # At least some Sender params must have non-zero gradients
    sender_nonzero = [n for n, v in sender_grad_norms.items() if v > 0]
    assert len(sender_nonzero) > 0, (
        "No Sender parameters have non-zero gradients after train_step. "
        "REINFORCE signal is not flowing back to the Sender."
    )

    # At least some Receiver params must have non-zero gradients
    receiver_nonzero = [n for n, v in receiver_grad_norms.items() if v > 0]
    assert len(receiver_nonzero) > 0, (
        "No Receiver parameters have non-zero gradients after train_step."
    )


# ---------------------------------------------------------------------------
# Test 3: EMA baseline updates over multiple steps
# ---------------------------------------------------------------------------

def test_baseline_updates(models, batch):
    """
    The moving-average baseline must change over multiple steps.
    A baseline stuck at its initial value indicates the EMA update is broken.
    """
    sender, receiver = models
    optimizer = torch.optim.Adam(
        list(sender.parameters()) + list(receiver.parameters()), lr=0.01
    )

    baseline = BASELINE_INIT
    baselines_seen = [baseline]

    for _ in range(10):
        _, _, _, baseline = train_step(
            sender, receiver, batch, optimizer, baseline, BASELINE_DECAY
        )
        baselines_seen.append(baseline)

    print(f"\nBaseline trajectory: {[f'{b:.4f}' for b in baselines_seen]}")

    # After 10 steps the baseline must have moved from its initial value
    assert baselines_seen[-1] != BASELINE_INIT, (
        f"Baseline did not update from initial value {BASELINE_INIT} "
        f"after 10 steps. Got: {baselines_seen}"
    )

    # It should not be NaN
    assert not torch.isnan(torch.tensor(baselines_seen[-1])), "Baseline is NaN"
