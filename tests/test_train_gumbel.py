"""
tests/test_train_gumbel.py
-----------------------------------------------------------------------------
Tests for Phase 3 Gumbel-Softmax training logic.
"""

from __future__ import annotations

import os
import random
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.synthetic_generator import ConceptSpace, sample_batch
from src.sender import Sender
from src.receiver import Receiver
from src.train_gumbel import gumbel_generate, train_step

# Small settings for fast tests
VOCAB_SIZES = (2, 3)
CONCEPT_DIM = sum(VOCAB_SIZES) # 5
VOCAB_SIZE = 5
EMBED_DIM = 8
HIDDEN_DIM = 16
MAX_LENGTH = 3
BATCH_SIZE = 4
N_DISTRACTORS = 2


@pytest.fixture
def models():
    torch.manual_seed(42)
    sender = Sender(CONCEPT_DIM, VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, MAX_LENGTH)
    receiver = Receiver(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, CONCEPT_DIM)
    return sender, receiver

@pytest.fixture
def batch():
    rng = random.Random(42)
    cs = ConceptSpace(VOCAB_SIZES)
    return sample_batch(
        BATCH_SIZE,
        vocab_sizes=VOCAB_SIZES,
        n_distractors=N_DISTRACTORS,
        rng=rng,
        concept_space=cs,
    )

def test_gumbel_generate_shape(models, batch):
    """gumbel_generate produces the correct shape soft message."""
    sender, _ = models
    targets = batch["target_vectors"]
    soft_msg = gumbel_generate(sender, targets, temperature=1.0)
    assert soft_msg.shape == (BATCH_SIZE, MAX_LENGTH, VOCAB_SIZE)
    # Check that it's differentiable (requires_grad is True)
    assert soft_msg.requires_grad

def test_train_step_finite_loss(models, batch):
    """One training step runs without error and produces a finite loss."""
    sender, receiver = models
    optimizer = torch.optim.Adam(
        list(sender.parameters()) + list(receiver.parameters()), lr=0.01
    )
    loss, acc = train_step(sender, receiver, batch, optimizer)
    
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert not torch.isnan(torch.tensor(loss))
    assert not torch.isinf(torch.tensor(loss))
    assert 0.0 <= acc <= 1.0

def test_gradient_flow_to_sender(models, batch):
    """
    Gradients must reach the Sender's early layers (e.g., concept_encoder) 
    after a full Gumbel-Softmax pass.
    """
    sender, receiver = models
    optimizer = torch.optim.Adam(
        list(sender.parameters()) + list(receiver.parameters()), lr=0.01
    )
    
    train_step(sender, receiver, batch, optimizer)
    
    # Check a few key sender parameters to ensure gradients flowed all the way back
    # concept_encoder is the earliest layer
    for name, param in sender.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient!"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient!"
        assert param.grad.norm().item() > 0, f"Parameter {name} has zero gradient!"
        
def test_loss_decreases_on_fixed_batch(models, batch):
    """
    Sanity check: over a small number of steps on a FIXED batch,
    the loss should decrease, indicating that learning is happening.
    """
    sender, receiver = models
    optimizer = torch.optim.Adam(
        list(sender.parameters()) + list(receiver.parameters()), lr=0.05
    )
    
    # Run first step
    initial_loss, _ = train_step(sender, receiver, batch, optimizer)
    
    # Run 50 more steps on the SAME batch
    for _ in range(50):
        loss, acc = train_step(sender, receiver, batch, optimizer)
        
    final_loss = loss
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
    # Accuracy on a fixed batch should ideally reach 1.0 or get very close
    assert acc > 0.5, f"Model failed to overfit tiny batch; acc={acc}"

from unittest.mock import patch
from src.train_gumbel import train, DEFAULT_CONFIG
import copy
import torch

def test_early_stopping_and_checkpoint_restore(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["training"]["epochs"] = 20
    config["training"]["patience"] = 3
    config["logging"]["log_dir"] = str(tmp_path)
    
    # Track the epochs and validation losses
    val_losses = [1.5, 1.4, 1.3, 1.35, 1.4, 1.5, 1.6] # Minimum at epoch 3. Should stop at epoch 6 (patience 3: 4, 5, 6)
    
    # Mock F.cross_entropy to return these specific values during eval, but let it do its thing during train_step
    # Wait, train_step calls F.cross_entropy too!
    # Instead, let's patch train_step to just do nothing and return (1.0, 1.0)
    # And patch the eval block's F.cross_entropy
    
    eval_call_count = 0
    def mock_eval_ce(*args, **kwargs):
        nonlocal eval_call_count
        if eval_call_count < len(val_losses):
            loss = val_losses[eval_call_count]
        else:
            loss = 2.0
        eval_call_count += 1
        return torch.tensor(loss)
        
    with patch("src.train_gumbel.train_step", return_value=(1.0, 1.0)):
        with patch("torch.nn.functional.cross_entropy", side_effect=mock_eval_ce):
            # Also patch sender.load_state_dict to prove it was called
            with patch("src.sender.Sender.load_state_dict") as mock_load:
                train(config)
                
    # Training should have stopped at epoch 6 because minimum was at epoch 3, patience is 3 (epochs 4, 5, 6 no improvement).
    assert eval_call_count == 6
    mock_load.assert_called_once()

def test_temperature_annealing(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["training"]["epochs"] = 3
    config["gumbel"]["start_temperature"] = 2.0
    config["gumbel"]["decay_rate"] = 0.5
    config["gumbel"]["end_temperature"] = 0.1
    config["logging"]["log_dir"] = str(tmp_path)
    
    temps_seen = []
    
    original_train_step = train_step
    def mock_train_step(sender, receiver, batch, optimizer, temperature):
        temps_seen.append(temperature)
        return (1.0, 1.0)
        
    with patch("src.train_gumbel.train_step", side_effect=mock_train_step):
        train(config)
        
    # Epoch 1: 2.0 * (0.5**0) = 2.0
    # Epoch 2: 2.0 * (0.5**1) = 1.0
    # Epoch 3: 2.0 * (0.5**2) = 0.5
    assert len(temps_seen) == 3
    assert abs(temps_seen[0] - 2.0) < 1e-5
    assert abs(temps_seen[1] - 1.0) < 1e-5
    assert abs(temps_seen[2] - 0.5) < 1e-5
