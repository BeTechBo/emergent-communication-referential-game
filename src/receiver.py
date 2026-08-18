"""
src/receiver.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 – Receiver agent (architecture only, no training).

The Receiver reads the Sender's message and scores each candidate concept,
picking the one most consistent with the message.

Architecture
────────────
1. Symbol projection layer:  vocab_size  →  embed_dim   (bias-free Linear)
   Accepts *either* hard one-hot vectors (REINFORCE path in Phase 3) or soft
   probability / Gumbel-Softmax distributions — both are (*, vocab_size) shaped,
   so a single linear layer handles both without any branching.

   For discrete indices (B, L) the forward() method converts them to one-hot
   (B, L, vocab_size) first, so the same projection applies transparently.

2. LSTM message encoder:  embed_dim  →  hidden_dim   (full nn.LSTM, batch-first)
   Reads the projected symbol sequence; the final hidden state h_n serves as
   the message embedding.

3. Concept projector:  concept_dim  →  hidden_dim   (Linear + ReLU)
   Projects every candidate concept vector into the same hidden_dim space as
   the message embedding.

4. Dot-product scorer:
   score_i  =  msg_emb  ·  cand_emb_i
   Produces one raw scalar per candidate, returned as (B, n_candidates).
   Phase 3 applies softmax / cross-entropy over these scores.

Input flexibility (Phase-3 requirement)
────────────────────────────────────────
forward(message, candidates) accepts:
  • message as (B, L) int64      — discrete symbol indices (REINFORCE)
  • message as (B, L, vocab_size) float32 — soft distributions (Gumbel-Softmax)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Receiver(nn.Module):
    """
    Receiver agent: (message, candidate concepts) → per-candidate scores.

    Parameters
    ----------
    vocab_size : int
        Size of the communication vocabulary (must match Sender.vocab_size).
    embed_dim : int
        Dimensionality of the symbol embedding fed into the LSTM.
    hidden_dim : int
        Dimensionality of the LSTM hidden state and the concept projection.
    concept_dim : int
        Dimensionality of each input concept vector (= sum of Phase-1
        attribute vocab sizes, e.g. 4+4+5 = 13).
    n_lstm_layers : int, optional
        Number of stacked LSTM layers (default 1).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        concept_dim: int,
        n_lstm_layers: int = 1,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.concept_dim = concept_dim

        # ── 1. Symbol projection (bias-free → equivalent to embedding lookup
        #       when input is one-hot, but differentiable w.r.t. soft inputs) ─
        self.symbol_proj = nn.Linear(vocab_size, embed_dim, bias=False)

        # ── 2. LSTM message encoder ─────────────────────────────────────────
        self.message_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
        )

        # ── 3. Concept projector ────────────────────────────────────────────
        self.concept_proj = nn.Sequential(
            nn.Linear(concept_dim, hidden_dim),
            nn.ReLU(),
        )

        # Weight initialisation
        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _project_message(self, message: torch.Tensor) -> torch.Tensor:
        """
        Convert a message (discrete or soft) to a sequence of embeddings.

        Parameters
        ----------
        message : (B, L) int64   — discrete symbol indices
                  or (B, L, vocab_size) float32 — soft distributions

        Returns
        -------
        (B, L, embed_dim) float32
        """
        if message.dim() == 2:
            # Discrete path: convert indices → one-hot, then project.
            # Using F.one_hot so the one-hot itself is non-differentiable
            # (as expected for the REINFORCE path in Phase 3).
            one_hot = F.one_hot(message, num_classes=self.vocab_size).float()
            # one_hot: (B, L, vocab_size)
            return self.symbol_proj(one_hot)   # (B, L, embed_dim)
        elif message.dim() == 3:
            # Soft path: (B, L, vocab_size) — directly project.
            return self.symbol_proj(message)   # (B, L, embed_dim)
        else:
            raise ValueError(
                f"message must be 2-D (discrete) or 3-D (soft), "
                f"got {message.dim()}-D tensor."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        message: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score each candidate concept given the received message.

        Parameters
        ----------
        message : (B, L) int64
            Sequence of discrete symbol indices (REINFORCE path in Phase 3).
            OR
            (B, L, vocab_size) float32
            Soft symbol distributions (Gumbel-Softmax path in Phase 3).

        candidates : (B, n_candidates, concept_dim) float32
            The full candidate set for each batch element
            (target + distractors in shuffled order, as produced by
            sample_batch's ``"all_candidates"`` key).

        Returns
        -------
        scores : (B, n_candidates) float32
            Raw (un-normalised) dot-product scores, one per candidate.
            Phase 3 applies softmax / cross-entropy over these.
        """
        # ── Step 1: project message symbols → embedding sequence ────────────
        msg_emb_seq = self._project_message(message)  # (B, L, embed_dim)

        # ── Step 2: LSTM encode → take final hidden state ───────────────────
        _, (h_n, _) = self.message_lstm(msg_emb_seq)
        # h_n shape: (n_lstm_layers, B, hidden_dim)
        # Take the top layer's hidden state as the message embedding.
        msg_emb = h_n[-1]                             # (B, hidden_dim)

        # ── Step 3: project candidate concept vectors ────────────────────────
        cand_emb = self.concept_proj(candidates)      # (B, n_candidates, hidden_dim)

        # ── Step 4: dot-product score per candidate ──────────────────────────
        # Expand msg_emb for batch matrix multiply:
        #   msg_emb.unsqueeze(-1): (B, hidden_dim, 1)
        #   bmm result:            (B, n_candidates, 1) → squeeze → (B, n_candidates)
        scores = torch.bmm(cand_emb, msg_emb.unsqueeze(-1)).squeeze(-1)

        return scores                                 # (B, n_candidates)
