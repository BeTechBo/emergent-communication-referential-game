"""
src/sender.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 – Sender agent (architecture only, no training).

The Sender observes a target concept vector and encodes it as a short sequence
of symbol logits.  It does NOT sample or discretize the symbols here — that
responsibility belongs to Phase 3's training scripts (Gumbel-Softmax or
REINFORCE), which will call into the step-level API defined below.

Architecture
────────────
1. MLP concept encoder:  concept_dim  →  hidden_dim
   Maps the target concept vector to initial LSTM hidden/cell states.

2. LSTM decoder (LSTMCell, one step at a time):
   embed_dim  →  hidden_dim
   Runs for max_length steps.

3. Output projection:  hidden_dim  →  vocab_size
   Produces raw logits over the symbol vocabulary at each timestep.

Symbol vocabulary convention
─────────────────────────────
Index 0 is reserved for EOS (end-of-sequence).  The architecture emits logits
for all vocab_size symbols (including EOS) at every step; early-stopping logic
is left for Phase 3.

Phase-3 integration points
───────────────────────────
encode(concept)          → (h_0, c_0)
    Encode concept to initial LSTM state; call once per batch.

sos_input(batch_size)    → (B, embed_dim)
    Returns the learnable start-of-sequence embedding expanded to B.

step(inp, h, c)          → (logits, h', c')
    Single LSTM decode step; Phase 3 loops over this, inserting its own
    symbol selection (Gumbel-Softmax relaxation or REINFORCE sample) between
    successive calls.

forward(concept)         → (B, max_length, vocab_size)
    Convenience wrapper for shape testing and inference: runs all max_length
    steps with zero feedback (no autoregressive loop), used in Phase 2 tests.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class Sender(nn.Module):
    """
    Sender agent: concept vector → sequence of symbol logits.

    Parameters
    ----------
    concept_dim : int
        Dimensionality of the input concept vector (= sum of Phase-1 vocab
        sizes, e.g. 4+4+5 = 13 for the default concept space).
    vocab_size : int
        Number of symbols in the shared communication vocabulary.
        Index 0 is reserved for EOS; effective content symbols are 1..vocab_size-1.
    embed_dim : int
        Dimensionality of the symbol embedding space fed into the LSTM.
    hidden_dim : int
        Dimensionality of the LSTM hidden state (and cell state).
    max_length : int
        Maximum number of symbols the Sender may produce per message.
    mlp_hidden_dim : int, optional
        Width of the hidden layer in the MLP concept encoder.
        Defaults to ``hidden_dim``.
    """

    def __init__(
        self,
        concept_dim: int,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        max_length: int,
        mlp_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        if mlp_hidden_dim is None:
            mlp_hidden_dim = hidden_dim

        self.concept_dim = concept_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # ── 1. MLP concept encoder ──────────────────────────────────────────
        # Maps concept_dim → hidden_dim, then projected to h_0 and c_0
        # separately so the two LSTM state components are independently learned.
        self.concept_encoder = nn.Sequential(
            nn.Linear(concept_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.h0_proj = nn.Linear(hidden_dim, hidden_dim)
        self.c0_proj = nn.Linear(hidden_dim, hidden_dim)

        # ── 2. Learnable start-of-sequence (SOS) embedding ─────────────────
        # Shape (1, embed_dim); expanded to (B, embed_dim) at forward time.
        self.sos = nn.Parameter(torch.zeros(1, embed_dim))

        # ── 3. Symbol embedding ─────────────────────────────────────────────
        # Used by Phase 3 (REINFORCE path) to embed discrete sampled tokens
        # before feeding them back as the next LSTM input.
        # In Phase 2 forward() we do not use this (zero-feedback pass).
        self.symbol_embedding = nn.Embedding(vocab_size, embed_dim)

        # ── 4. LSTM decoder (cell-level for step-by-step control) ───────────
        self.lstm_cell = nn.LSTMCell(embed_dim, hidden_dim)

        # ── 5. Output projection ────────────────────────────────────────────
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Weight initialisation
        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers; orthogonal for LSTM weights."""
        for name, p in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        # SOS starts at zero — a neutral starting point.
        nn.init.zeros_(self.sos)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase-3 integration API
    # ──────────────────────────────────────────────────────────────────────────

    def encode(self, concept: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of concept vectors into an initial LSTM state.

        Parameters
        ----------
        concept : (B, concept_dim) float32

        Returns
        -------
        h_0 : (B, hidden_dim) float32
        c_0 : (B, hidden_dim) float32
        """
        enc = self.concept_encoder(concept)      # (B, hidden_dim)
        h_0 = self.h0_proj(enc)                  # (B, hidden_dim)
        c_0 = self.c0_proj(enc)                  # (B, hidden_dim)
        return h_0, c_0

    def sos_input(self, batch_size: int) -> torch.Tensor:
        """
        Return the SOS embedding expanded to the batch dimension.

        Returns
        -------
        (B, embed_dim) float32  (shares storage with self.sos for gradient flow)
        """
        return self.sos.expand(batch_size, -1)   # (B, embed_dim)

    def step(
        self,
        inp: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One LSTM decoding step.

        Parameters
        ----------
        inp : (B, embed_dim) float32
            Input embedding for this timestep (SOS at t=0; symbol embedding
            or Gumbel-Softmax relaxation at t>0 — provided by Phase 3).
        h   : (B, hidden_dim) float32  — previous hidden state
        c   : (B, hidden_dim) float32  — previous cell state

        Returns
        -------
        logits : (B, vocab_size) float32  — raw unnormalised symbol scores
        h_new  : (B, hidden_dim) float32
        c_new  : (B, hidden_dim) float32
        """
        h_new, c_new = self.lstm_cell(inp, (h, c))
        logits = self.output_proj(h_new)          # (B, vocab_size)
        return logits, h_new, c_new

    # ──────────────────────────────────────────────────────────────────────────
    # Phase-2 convenience forward pass
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, concept: torch.Tensor) -> torch.Tensor:
        """
        Full message generation — architecture / shape testing only.

        Runs max_length LSTM steps using the SOS embedding at t=0 and
        zero-vector feedback at t>0 (no autoregressive symbol selection).
        Phase 3 replaces this loop with Gumbel-Softmax or REINFORCE sampling
        via repeated calls to encode() / sos_input() / step().

        Parameters
        ----------
        concept : (B, concept_dim) float32

        Returns
        -------
        logits : (B, max_length, vocab_size) float32
            Raw per-timestep symbol scores; NOT normalised / softmax'd.
        """
        B = concept.size(0)
        h, c = self.encode(concept)              # (B, hidden_dim) each
        inp = self.sos_input(B)                  # (B, embed_dim)

        logits_list = []
        for t in range(self.max_length):
            logits, h, c = self.step(inp, h, c)
            logits_list.append(logits)           # (B, vocab_size)

            # Zero-feedback: no symbol sampling; Phase 3 overrides this.
            inp = torch.zeros(B, self.embed_dim, device=concept.device,
                              dtype=concept.dtype)

        return torch.stack(logits_list, dim=1)   # (B, max_length, vocab_size)
