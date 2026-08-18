"""
tests/test_agents.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 unit tests for src/sender.py and src/receiver.py.

Coverage
─────────
Sender
  1. Output shape is (B, max_length, vocab_size).
  2. Output dtype is float32.
  3. Output is different for different concept inputs (module responds to input).
  4a. Gradients flow through every parameter that forward() uses.
  4b. symbol_embedding gets gradients when used via the step() API (REINFORCE
      style) — verifying it is wired correctly for Phase 3.
  5. encode() returns tensors of the right shape.
  6. step() returns (logits, h, c) of the right shapes.
  7. sos_input() shape matches (B, embed_dim).
  8. Configurable hyperparameters are reflected in output shapes.

Receiver
  9.  Output shape is (B, n_candidates) for discrete-index messages.
  10. Output dtype is float32 for discrete-index messages.
  11. Output shape is (B, n_candidates) for soft-distribution messages.
  12. Output dtype is float32 for soft-distribution messages.
  13. Discrete and soft paths produce the same output for one-hot soft input.
  14. Gradients flow through every parameter (discrete message path).
  15. Gradients flow through every parameter (soft message path, end-to-end diff).
  16. Configurable hyperparameters are reflected in output shapes.

Integration
  17. Sender → Receiver pipeline: shapes and gradient flow together.

Design note – symbol_embedding
───────────────────────────────
Sender.symbol_embedding is intentionally NOT exercised by Sender.forward(),
which uses a zero-feedback pass (no autoregressive loop) for Phase 2 shape
testing.  symbol_embedding is a Phase-3 utility: the REINFORCE training script
embeds discrete sampled tokens and feeds them back into Sender.step().
Gradient tests for the zero-feedback forward() therefore exclude this parameter
and instead verify it separately via the step() API (test 4b below).
"""

from __future__ import annotations

import sys
import os

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.sender import Sender
from src.receiver import Receiver

# ─────────────────────────────────────────────────────────────────────────────
# Shared defaults (kept small for fast tests)
# ─────────────────────────────────────────────────────────────────────────────

CONCEPT_DIM  = 13   # 4+4+5 (Phase-1 default)
VOCAB_SIZE   = 10   # includes EOS at index 0
EMBED_DIM    = 16
HIDDEN_DIM   = 32
MAX_LENGTH   = 5
N_CANDIDATES = 5    # 1 target + 4 distractors
BATCH_SIZE   = 8


def make_sender(**kwargs) -> Sender:
    defaults = dict(
        concept_dim=CONCEPT_DIM,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        max_length=MAX_LENGTH,
    )
    defaults.update(kwargs)
    return Sender(**defaults)


def make_receiver(**kwargs) -> Receiver:
    defaults = dict(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        concept_dim=CONCEPT_DIM,
    )
    defaults.update(kwargs)
    return Receiver(**defaults)


def random_concepts(B: int = BATCH_SIZE) -> torch.Tensor:
    """Dummy concept vectors — random floats simulating one-hot encodings."""
    return torch.randn(B, CONCEPT_DIM)


def discrete_message(B: int = BATCH_SIZE, L: int = MAX_LENGTH) -> torch.Tensor:
    """Random discrete symbol indices in [0, VOCAB_SIZE)."""
    return torch.randint(0, VOCAB_SIZE, (B, L))


def soft_message(B: int = BATCH_SIZE, L: int = MAX_LENGTH) -> torch.Tensor:
    """Random soft symbol distributions (sum to 1 per step)."""
    return F.softmax(torch.randn(B, L, VOCAB_SIZE), dim=-1)


def random_candidates(B: int = BATCH_SIZE,
                      N: int = N_CANDIDATES) -> torch.Tensor:
    return torch.randn(B, N, CONCEPT_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: assert all parameters have .grad populated after backward()
# ─────────────────────────────────────────────────────────────────────────────

def assert_all_grads(
    module: torch.nn.Module,
    prefix: str = "",
    exclude: set | None = None,
) -> None:
    """
    Assert that every trainable parameter has a non-None, non-NaN gradient.

    Parameters
    ----------
    exclude : set of str, optional
        Parameter names (as returned by named_parameters()) to skip.
        Use this for parameters that are intentionally absent from a specific
        compute path (e.g. Sender.symbol_embedding in the zero-feedback forward).
    """
    if exclude is None:
        exclude = set()
    for name, p in module.named_parameters():
        if name in exclude:
            continue
        assert p.grad is not None, (
            f"{prefix}Parameter '{name}' has no gradient after backward()."
        )
        assert not torch.isnan(p.grad).any(), (
            f"{prefix}Parameter '{name}' has NaN gradient."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1-8  Sender tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSender:

    def test_output_shape(self):
        """1. forward() output shape is (B, max_length, vocab_size)."""
        sender = make_sender()
        logits = sender(random_concepts())
        assert logits.shape == (BATCH_SIZE, MAX_LENGTH, VOCAB_SIZE)

    def test_output_dtype(self):
        """2. forward() output dtype is float32."""
        sender = make_sender()
        logits = sender(random_concepts())
        assert logits.dtype == torch.float32

    def test_output_varies_with_input(self):
        """3. Different concept inputs produce different logits."""
        sender = make_sender()
        sender.eval()
        with torch.no_grad():
            c1 = torch.zeros(1, CONCEPT_DIM)
            c2 = torch.ones(1, CONCEPT_DIM)
            assert not torch.allclose(sender(c1), sender(c2))

    def test_gradient_flow(self):
        """
        4a. Gradients reach every parameter that forward() exercises.

        Sender.symbol_embedding is excluded here because forward() uses a
        zero-feedback pass (no autoregressive loop) — symbol_embedding is
        never called.  It is tested separately in test_symbol_embedding_grad.
        """
        # symbol_embedding is intentionally absent from the zero-feedback path.
        NOT_IN_FORWARD_PATH = {"symbol_embedding.weight"}

        sender = make_sender()
        sender.zero_grad()
        logits = sender(random_concepts())      # (B, L, V)
        loss = logits.sum()
        loss.backward()
        assert_all_grads(sender, prefix="Sender: ", exclude=NOT_IN_FORWARD_PATH)

    def test_symbol_embedding_grad(self):
        """
        4b. symbol_embedding gets gradients when used via the step() API.

        This simulates the Phase-3 REINFORCE path: sample a discrete token,
        embed it with symbol_embedding, and feed it as the next step input.
        Confirms the embedding is properly wired into the compute graph.
        """
        sender = make_sender()
        sender.zero_grad()
        B = BATCH_SIZE

        h, c = sender.encode(random_concepts(B))
        inp = sender.sos_input(B)

        logits_list = []
        for t in range(MAX_LENGTH):
            logits, h, c = sender.step(inp, h, c)
            logits_list.append(logits)
            # Simulate REINFORCE: argmax (non-differentiable) then embed.
            # We use argmax (detached) so this mirrors the actual Phase-3 path.
            sampled = logits.detach().argmax(dim=-1)       # (B,)
            inp = sender.symbol_embedding(sampled)         # (B, embed_dim)

        loss = torch.stack(logits_list, dim=1).sum()
        loss.backward()

        # symbol_embedding rows that were looked up must have gradients.
        assert sender.symbol_embedding.weight.grad is not None, (
            "symbol_embedding.weight has no gradient after step()-based loop."
        )


    def test_encode_shapes(self):
        """5. encode() returns h and c of shape (B, hidden_dim)."""
        sender = make_sender()
        h, c = sender.encode(random_concepts())
        assert h.shape == (BATCH_SIZE, HIDDEN_DIM)
        assert c.shape == (BATCH_SIZE, HIDDEN_DIM)

    def test_step_shapes(self):
        """6. step() returns (logits, h, c) with correct shapes."""
        sender = make_sender()
        B = BATCH_SIZE
        h, c = sender.encode(random_concepts(B))
        inp = sender.sos_input(B)
        logits, h2, c2 = sender.step(inp, h, c)
        assert logits.shape == (B, VOCAB_SIZE)
        assert h2.shape == (B, HIDDEN_DIM)
        assert c2.shape == (B, HIDDEN_DIM)

    def test_sos_input_shape(self):
        """7. sos_input() shape is (B, embed_dim)."""
        sender = make_sender()
        sos = sender.sos_input(BATCH_SIZE)
        assert sos.shape == (BATCH_SIZE, EMBED_DIM)

    def test_custom_hyperparameters(self):
        """8. Custom hyperparameters are reflected in output shape."""
        vocab_size_  = 20
        max_length_  = 7
        hidden_dim_  = 64
        embed_dim_   = 24
        sender = make_sender(
            vocab_size=vocab_size_,
            max_length=max_length_,
            hidden_dim=hidden_dim_,
            embed_dim=embed_dim_,
        )
        logits = sender(random_concepts())
        assert logits.shape == (BATCH_SIZE, max_length_, vocab_size_)


# ─────────────────────────────────────────────────────────────────────────────
# 9-16  Receiver tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReceiver:

    def test_output_shape_discrete(self):
        """9. Output shape is (B, n_candidates) for discrete messages."""
        receiver = make_receiver()
        scores = receiver(discrete_message(), random_candidates())
        assert scores.shape == (BATCH_SIZE, N_CANDIDATES)

    def test_output_dtype_discrete(self):
        """10. Output dtype is float32 for discrete messages."""
        receiver = make_receiver()
        scores = receiver(discrete_message(), random_candidates())
        assert scores.dtype == torch.float32

    def test_output_shape_soft(self):
        """11. Output shape is (B, n_candidates) for soft messages."""
        receiver = make_receiver()
        scores = receiver(soft_message(), random_candidates())
        assert scores.shape == (BATCH_SIZE, N_CANDIDATES)

    def test_output_dtype_soft(self):
        """12. Output dtype is float32 for soft messages."""
        receiver = make_receiver()
        scores = receiver(soft_message(), random_candidates())
        assert scores.dtype == torch.float32

    def test_soft_onehot_equals_discrete(self):
        """
        13. When the soft message is a hard one-hot (i.e. equivalent to
        a discrete index), the Receiver must produce the same scores
        as the discrete path.  This verifies that symbol_proj is the
        canonical shared projection for both paths.
        """
        receiver = make_receiver()
        receiver.eval()

        # Build discrete indices and their exact one-hot equivalents.
        msg_disc = torch.tensor([[2, 5, 0, 7, 3]])    # (1, L)
        msg_soft = F.one_hot(msg_disc, VOCAB_SIZE).float()  # (1, L, V)
        cands = random_candidates(B=1)

        with torch.no_grad():
            scores_d = receiver(msg_disc, cands)
            scores_s = receiver(msg_soft, cands)

        assert torch.allclose(scores_d, scores_s, atol=1e-5), (
            "Discrete and equivalent one-hot soft scores differ."
        )

    def test_gradient_flow_discrete(self):
        """14. Gradients flow through all parameters (discrete path)."""
        receiver = make_receiver()
        receiver.zero_grad()
        scores = receiver(discrete_message(), random_candidates())
        scores.sum().backward()
        assert_all_grads(receiver, prefix="Receiver (discrete): ")

    def test_gradient_flow_soft(self):
        """
        15. Gradients flow through all parameters (soft path).
        Unlike the discrete path, the soft path is fully differentiable
        end-to-end — this is the Gumbel-Softmax training mode.
        """
        receiver = make_receiver()
        receiver.zero_grad()
        msg = soft_message()
        # Keep message in computation graph (requires_grad for the soft tensor).
        msg.requires_grad_(True)
        scores = receiver(msg, random_candidates())
        scores.sum().backward()
        assert_all_grads(receiver, prefix="Receiver (soft): ")
        # Also verify gradient w.r.t. the soft message itself.
        assert msg.grad is not None, "No gradient reached the soft message tensor."

    def test_custom_hyperparameters(self):
        """16. Custom hyperparameters are reflected in output shape."""
        n_cands  = 8
        h_dim    = 64
        v_size   = 15
        receiver = make_receiver(
            vocab_size=v_size,
            hidden_dim=h_dim,
            n_lstm_layers=2,
        )
        msg   = torch.randint(0, v_size, (BATCH_SIZE, MAX_LENGTH))
        cands = random_candidates(N=n_cands)
        scores = receiver(msg, cands)
        assert scores.shape == (BATCH_SIZE, n_cands)


# ─────────────────────────────────────────────────────────────────────────────
# 17  Integration: Sender → Receiver pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestSenderReceiverPipeline:

    def test_pipeline_shapes(self):
        """
        17a. Sender logits feed into Receiver via soft path:
        end-to-end shapes are correct.
        """
        sender   = make_sender()
        receiver = make_receiver()
        concepts = random_concepts()
        cands    = random_candidates()

        sender_logits = sender(concepts)               # (B, L, V)
        # Treat Sender logits as soft message (Gumbel-Softmax will do this).
        scores = receiver(sender_logits, cands)        # (B, N_CANDS)

        assert sender_logits.shape == (BATCH_SIZE, MAX_LENGTH, VOCAB_SIZE)
        assert scores.shape         == (BATCH_SIZE, N_CANDIDATES)

    def test_pipeline_gradient_flow(self):
        """
        17b. Gradients flow through Sender and Receiver jointly when connected
        via the soft message path (simulating Gumbel-Softmax training).

        Sender.symbol_embedding is excluded for the same reason as in
        TestSender.test_gradient_flow: forward() is zero-feedback and never
        calls symbol_embedding.  All other Sender and all Receiver parameters
        must receive gradients.
        """
        SENDER_EXCLUDE = {"symbol_embedding.weight"}

        sender   = make_sender()
        receiver = make_receiver()

        sender.zero_grad()
        receiver.zero_grad()

        concepts = random_concepts()
        cands    = random_candidates()

        sender_logits = sender(concepts)               # (B, L, V)
        soft_msg      = F.softmax(sender_logits, dim=-1)  # (B, L, V)
        scores        = receiver(soft_msg, cands)      # (B, N_CANDS)

        # Dummy loss: maximise score of a fixed "correct" candidate.
        target_idx = torch.zeros(BATCH_SIZE, dtype=torch.long)
        loss = F.cross_entropy(scores, target_idx)
        loss.backward()

        assert_all_grads(sender,   prefix="Pipeline/Sender:   ", exclude=SENDER_EXCLUDE)
        assert_all_grads(receiver, prefix="Pipeline/Receiver: ")


    def test_pipeline_discrete_receiver(self):
        """
        17c. Sender logits → argmax → Receiver discrete path (REINFORCE style).
        Gradients reach Receiver parameters (but not Sender, as expected for
        the non-differentiable discrete step — no assertion for Sender here).
        """
        sender   = make_sender()
        receiver = make_receiver()
        receiver.zero_grad()

        concepts = random_concepts()
        cands    = random_candidates()

        with torch.no_grad():
            sender_logits = sender(concepts)           # (B, L, V)

        # Hard discrete message (no grad through Sender).
        disc_msg = sender_logits.argmax(dim=-1)        # (B, L) int64
        scores   = receiver(disc_msg, cands)

        scores.sum().backward()
        assert_all_grads(receiver, prefix="Pipeline/Receiver (discrete): ")
