"""
src/metrics.py
-----------------------------------------------------------------------------
Phase 5 - Analysis of emergent communication protocols.

Contains functions to measure:
1. Message entropy (how much of the message space is utilized).
2. Topographic similarity (correlation between concept distance and message distance).
3. Symbol usage (correlation between specific symbols and concept attributes).
"""

from __future__ import annotations

import collections
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import scipy.stats

from src.sender import Sender


def _get_discrete_messages(sender: Sender, concepts: torch.Tensor) -> torch.Tensor:
    """
    Get discrete argmax messages for a batch of concepts.
    Returns: (B, max_length) int64
    """
    sender.eval()
    with torch.no_grad():
        # sender(concepts) returns (B, L, vocab_size) logits
        logits = sender(concepts)
        return logits.argmax(dim=-1)


def message_entropy(sender: Sender, all_concepts: torch.Tensor) -> float:
    """
    Compute the entropy of the empirical distribution over unique messages
    generated for the full concept set.

    H = -sum( p(m) * log2(p(m)) )
    """
    messages = _get_discrete_messages(sender, all_concepts)
    # Convert rows to tuples for counting
    msg_tuples = [tuple(m.tolist()) for m in messages]
    
    counts = collections.Counter(msg_tuples)
    total = len(msg_tuples)
    
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
        
    return entropy


def levenshtein_distance(seq1: List[int], seq2: List[int]) -> int:
    """Standard Edit Distance between two sequences."""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
            
    return dp[n][m]


def topographic_similarity(sender: Sender, all_concepts: torch.Tensor) -> float:
    """
    Compute the topographic similarity between the concept space and the
    emergent message space.
    
    Metric: Spearman rank correlation between pairwise distances.
    - Concept distance: Hamming distance (number of differing attributes).
    - Message distance: Levenshtein (edit) distance.
    
    Returns a float in [-1, 1]. Close to 1 means highly compositional.
    """
    messages = _get_discrete_messages(sender, all_concepts)
    
    n = all_concepts.size(0)
    concept_dists = []
    message_dists = []
    
    c_np = all_concepts.cpu().numpy()
    m_np = messages.cpu().numpy()
    
    # Pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            # Concept Hamming distance
            # For concatenated one-hots, sum of absolute differences divided by 2
            # is the number of attributes that differ.
            cdist = np.sum(np.abs(c_np[i] - c_np[j])) / 2.0
            
            # Message edit distance
            mdist = levenshtein_distance(m_np[i].tolist(), m_np[j].tolist())
            
            concept_dists.append(cdist)
            message_dists.append(mdist)
            
    # Spearman rank correlation (standard metric in emergent comms literature)
    corr, _ = scipy.stats.spearmanr(concept_dists, message_dists)
    
    if np.isnan(corr):
        return 0.0
    return float(corr)


def symbol_usage(sender: Sender, all_concepts: torch.Tensor, vocab_sizes: List[int]) -> List[str]:
    """
    Analyze whether specific symbols at specific positions correlate strongly
    with specific concept attribute values.
    
    Returns a list of human-readable string findings.
    """
    messages = _get_discrete_messages(sender, all_concepts)
    c_np = all_concepts.cpu().numpy()
    m_np = messages.cpu().numpy()
    
    N, L = m_np.shape
    
    # Convert one-hot concatenated concepts back to attribute indices
    attr_indices = []
    for row in c_np:
        idx_row = []
        offset = 0
        for vs in vocab_sizes:
            chunk = row[offset:offset+vs]
            val = int(np.argmax(chunk))
            idx_row.append(val)
            offset += vs
        attr_indices.append(idx_row)
    
    attr_indices = np.array(attr_indices)
    num_attrs = len(vocab_sizes)
    
    findings = []
    
    for pos in range(L):
        unique_syms = np.unique(m_np[:, pos])
        
        for sym in unique_syms:
            sym_mask = (m_np[:, pos] == sym)
            sym_count = sym_mask.sum()
            
            if sym_count < N * 0.05:
                continue
                
            for attr in range(num_attrs):
                attr_vals = attr_indices[sym_mask, attr]
                val_counts = collections.Counter(attr_vals)
                
                if not val_counts:
                    continue
                    
                most_common_val, count = val_counts.most_common(1)[0]
                
                p_attr_given_sym = count / sym_count
                
                total_val_count = (attr_indices[:, attr] == most_common_val).sum()
                if total_val_count == 0:
                    continue
                p_sym_given_attr = count / total_val_count
                
                if p_attr_given_sym >= 0.85 and p_sym_given_attr >= 0.85:
                    findings.append(
                        f"Symbol {sym} at pos {pos} strongly maps to Attr {attr}=Val {most_common_val} "
                        f"(P(attr|sym)={p_attr_given_sym:.2f}, P(sym|attr)={p_sym_given_attr:.2f})"
                    )
                    
    if not findings:
        findings.append("No strong, unambiguous symbol-attribute mappings found (the protocol may be highly entangled or positional).")
        
    return findings
