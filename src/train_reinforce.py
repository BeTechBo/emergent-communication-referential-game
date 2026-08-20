"""
src/train_reinforce.py
-----------------------------------------------------------------------------
Phase 4 - REINFORCE training for the emergent-communication referential game
(Havrylov & Titov, NeurIPS 2017).

Overview
--------
Unlike Phase 3 (Gumbel-Softmax), this script trains via the REINFORCE
policy-gradient estimator, meaning the Sender message is a sequence of
DISCRETE symbols sampled from categorical distributions. Because discrete
sampling is non-differentiable, we cannot back-propagate the Receiver loss
through the Sender directly. Instead we use the REINFORCE / Williams (1992)
policy gradient trick.

Gradient flow diagram
---------------------
Receiver path (fully differentiable):
  receiver_loss = cross_entropy(receiver_scores, target_indices)
  receiver_loss.backward()  -> Receiver parameters

Sender path (REINFORCE):
  For each timestep t the Sender emits logits -> we sample discrete symbol
  z_t ~ Categorical(logits_t). Sender loss:
    sender_loss = -mean_b [ sum_t [ log pi(z_t|state_t) ] * (reward_b - baseline) ]
  where:
    reward   = 1.0 if Receiver argmax == target_index, else 0.0
               (binary task success; cleaner comparison with Gumbel-Softmax
               than a shaped CE reward, since both methods see the same signal)
    baseline = exponential moving average of past rewards to reduce variance
  sender_loss.backward()    -> Sender parameters

Public API
----------
reinforce_generate(sender, concepts) -> (message, log_probs)
train_step(sender, receiver, batch, optimizer, baseline, baseline_decay)
    -> (sender_loss, receiver_loss, accuracy, new_baseline)
train(config) -> None
load_config(path) -> dict
"""

from __future__ import annotations

import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.synthetic_generator import ConceptSpace, sample_batch
from src.sender   import Sender
from src.receiver import Receiver


# -----------------------------------------------------------------------------
# Built-in default configuration
# -----------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "training": {
        "epochs": 400,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "patience": 30,
        "seed": 42,
    },
    "reinforce": {
        # EMA baseline: baseline_{t+1} = decay * baseline_t + (1-decay) * reward_t
        "baseline_decay": 0.99,
        # Start near chance level (1 / n_candidates = 0.2 for 5 candidates)
        "baseline_init": 0.2,
    },
    "model": {
        "vocab_size": 10,
        "embed_dim": 32,
        "hidden_dim": 64,
        "max_length": 5,
        "mlp_hidden_dim": 64,
    },
    "data": {
        "vocab_sizes": [6, 6, 8],
        "n_distractors": 4,
    },
    "logging": {
        "log_dir": "logs",
        "log_file": "train_reinforce.csv",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None = None) -> Dict[str, Any]:
    if path is None:
        return DEFAULT_CONFIG
    try:
        import yaml
        with open(path, "r") as fh:
            file_cfg = yaml.safe_load(fh) or {}
        return _deep_merge(DEFAULT_CONFIG, file_cfg)
    except ImportError:
        print("[train_reinforce] Warning: PyYAML not installed - using defaults.", file=sys.stderr)
        return DEFAULT_CONFIG
    except FileNotFoundError:
        print(f"[train_reinforce] Warning: config '{path}' not found - using defaults.", file=sys.stderr)
        return DEFAULT_CONFIG


# -----------------------------------------------------------------------------
# Discrete autoregressive message generation
# -----------------------------------------------------------------------------

def reinforce_generate(
    sender: Sender,
    concepts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Autoregressive discrete message generation via categorical sampling.

    At each timestep t:
      1. logits = sender.step(inp, h, c)
      2. dist   = Categorical(logits=logits)
      3. token  = dist.sample()          -- discrete, non-differentiable
      4. log_p  = dist.log_prob(token)   -- differentiable w.r.t. logits
      5. inp    = sender.symbol_embedding(token)  -- autoregressive feedback

    Gradients do not flow through the discrete sampling, but they DO flow
    through log_prob -> logits -> LSTM -> concept_encoder via the Sender loss.

    Returns
    -------
    message   : (B, max_length) int64
    log_probs : (B, max_length) float32
    """
    B = concepts.size(0)
    h, c = sender.encode(concepts)
    inp  = sender.sos_input(B)

    tokens: List[torch.Tensor]    = []
    log_ps: List[torch.Tensor]    = []

    for _ in range(sender.max_length):
        logits, h, c = sender.step(inp, h, c)
        dist    = Categorical(logits=logits)
        token   = dist.sample()
        log_p   = dist.log_prob(token)
        tokens.append(token)
        log_ps.append(log_p)
        inp = sender.symbol_embedding(token)

    return torch.stack(tokens, dim=1), torch.stack(log_ps, dim=1)


# -----------------------------------------------------------------------------
# One training step
# -----------------------------------------------------------------------------

def train_step(
    sender: Sender,
    receiver: Receiver,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    baseline: float,
    baseline_decay: float = 0.99,
) -> Tuple[float, float, float, float]:
    """
    One REINFORCE + cross-entropy update.

    Sender loss  (REINFORCE):
        L_sender = -mean_b [ sum_t log_pi_t * (reward_b - baseline) ]

    Receiver loss (standard CE):
        L_receiver = cross_entropy(scores, target_indices)

    Returns: (sender_loss, receiver_loss, accuracy, new_baseline)
    """
    targets        = batch["target_vectors"]
    candidates     = batch["all_candidates"]
    target_indices = batch["target_indices"]

    optimizer.zero_grad()

    # 1. Sender samples discrete message
    message, log_probs = reinforce_generate(sender, targets)

    # 2. Receiver scores candidates
    scores = receiver(message, candidates)

    # 3. Receiver CE loss (differentiable)
    receiver_loss = F.cross_entropy(scores, target_indices)

    # 4. Per-sample binary reward (no-grad)
    with torch.no_grad():
        predicted   = scores.argmax(dim=-1)
        reward      = (predicted == target_indices).float()
        mean_reward = reward.mean().item()

    # 5. REINFORCE sender loss
    log_prob_sum = log_probs.sum(dim=1)              # (B,)
    sender_loss  = -(log_prob_sum * (reward - baseline)).mean()

    # 6. Joint backward
    (sender_loss + receiver_loss).backward()
    optimizer.step()

    # 7. Update EMA baseline
    new_baseline = baseline_decay * baseline + (1 - baseline_decay) * mean_reward

    return sender_loss.item(), receiver_loss.item(), mean_reward, new_baseline


# -----------------------------------------------------------------------------
# ConceptSubspace helper
# -----------------------------------------------------------------------------

class ConceptSubspace(ConceptSpace):
    def __init__(self, vocab_sizes: List[int], subset: List[Tuple[int, ...]]) -> None:
        self.vocab_sizes    = tuple(vocab_sizes)
        self.n_attributes   = len(vocab_sizes)
        self.vector_dim     = sum(vocab_sizes)
        self._all_concepts  = subset
        self.n_concepts     = len(subset)


# -----------------------------------------------------------------------------
# Full training loop
# -----------------------------------------------------------------------------

def train(config: Dict[str, Any]) -> None:
    # -- Reproducibility -------------------------------------------------------
    seed = config["training"].get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    rng = random.Random(seed)

    # -- Hyperparameters -------------------------------------------------------
    epochs         = config["training"]["epochs"]
    batch_size     = config["training"]["batch_size"]
    lr             = config["training"]["learning_rate"]
    weight_decay   = config["training"]["weight_decay"]
    patience       = config["training"]["patience"]

    baseline_decay = config["reinforce"]["baseline_decay"]
    baseline_init  = config["reinforce"]["baseline_init"]

    vocab_sizes    = list(config["data"]["vocab_sizes"])
    n_distractors  = config["data"]["n_distractors"]
    concept_dim    = sum(vocab_sizes)

    vocab_size     = config["model"]["vocab_size"]
    embed_dim      = config["model"]["embed_dim"]
    hidden_dim     = config["model"]["hidden_dim"]
    max_length     = config["model"]["max_length"]
    mlp_hidden     = config["model"]["mlp_hidden_dim"]

    log_dir  = config["logging"]["log_dir"]
    log_file = config["logging"]["log_file"]

    # -- Concept space & split -------------------------------------------------
    cs = ConceptSpace(vocab_sizes)
    all_concepts = cs.all_concepts
    rng.shuffle(all_concepts)
    split_idx = int(len(all_concepts) * 0.8)
    train_cs = ConceptSubspace(vocab_sizes, all_concepts[:split_idx])
    val_cs   = ConceptSubspace(vocab_sizes, all_concepts[split_idx:])

    # -- Header ----------------------------------------------------------------
    tag = "[train_reinforce]"
    print(f"{tag} Concept space : {cs.n_concepts} total concepts")
    print(f"{tag} Train / Val   : {train_cs.n_concepts} / {val_cs.n_concepts}")
    print(f"{tag} Epochs        : {epochs} (patience={patience})")
    print(f"{tag} Batch size    : {batch_size}")
    print(f"{tag} LR            : {lr} (wd={weight_decay})")
    print(f"{tag} Baseline EMA  : decay={baseline_decay}, init={baseline_init}")
    print(f"{tag} Vocab sizes   : {vocab_sizes}  (concept_dim={concept_dim})")
    print(f"{tag} Msg vocab     : {vocab_size}  max_len={max_length}")
    print(f"{tag} Model config  : embed_dim={embed_dim}, hidden_dim={hidden_dim}, mlp_hidden_dim={mlp_hidden}")

    # -- Models ----------------------------------------------------------------
    sender   = Sender(concept_dim, vocab_size, embed_dim, hidden_dim,
                      max_length, mlp_hidden_dim=mlp_hidden)
    receiver = Receiver(vocab_size, embed_dim, hidden_dim, concept_dim)
    n_params = (sum(p.numel() for p in sender.parameters()) +
                sum(p.numel() for p in receiver.parameters()))
    print(f"{tag} Parameters    : {n_params:,}")

    # -- Optimizer -------------------------------------------------------------
    optimizer = torch.optim.Adam(
        list(sender.parameters()) + list(receiver.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    # -- Logging ---------------------------------------------------------------
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    csv_fh   = open(log_path, "w", newline="")
    writer   = csv.writer(csv_fh)
    writer.writerow(["epoch", "baseline", "sender_loss", "receiver_loss",
                     "train_acc", "val_loss", "val_acc"])
    print(f"{tag} Logging to    : {log_path}")
    print()

    col_w = 9
    header = (
        f"{'Epoch':>6}  {'Baseline':>8}  {'Snd Loss':>{col_w}}  "
        f"{'Rcv Loss':>{col_w}}  {'Trn Acc':>{col_w}}  {'Val Loss':>{col_w}}  {'Val Acc':>{col_w}}"
    )
    print(header)
    print("-" * len(header))

    best_val_loss       = float("inf")
    patience_counter    = 0
    best_sender_state   = None
    best_receiver_state = None
    best_epoch          = 0
    baseline            = float(baseline_init)

    t0 = time.time()
    total_steps = 0

    for epoch in range(1, epochs + 1):
        sender.train()
        receiver.train()

        steps_per_epoch = max(1, train_cs.n_concepts // batch_size)
        e_snd = e_rcv = e_acc = 0.0

        for _ in range(steps_per_epoch):
            batch = sample_batch(
                batch_size,
                vocab_sizes=vocab_sizes,
                n_distractors=n_distractors,
                rng=rng,
                concept_space=train_cs,
            )
            snd_l, rcv_l, acc, baseline = train_step(
                sender, receiver, batch, optimizer, baseline, baseline_decay
            )
            e_snd += snd_l
            e_rcv += rcv_l
            e_acc += acc
            total_steps += 1

        e_snd /= steps_per_epoch
        e_rcv /= steps_per_epoch
        e_acc /= steps_per_epoch

        # -- Validation --------------------------------------------------------
        sender.eval()
        receiver.eval()
        with torch.no_grad():
            val_bs  = min(batch_size, val_cs.n_concepts)
            vb      = sample_batch(
                val_bs, vocab_sizes=vocab_sizes, n_distractors=n_distractors,
                rng=rng, concept_space=val_cs,
            )
            val_msg, _ = reinforce_generate(sender, vb["target_vectors"])
            val_scores  = receiver(val_msg, vb["all_candidates"])
            val_loss    = F.cross_entropy(val_scores, vb["target_indices"]).item()
            val_acc     = (val_scores.argmax(-1) == vb["target_indices"]).float().mean().item()

        writer.writerow([epoch, f"{baseline:.4f}", f"{e_snd:.6f}", f"{e_rcv:.6f}",
                         f"{e_acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}"])
        csv_fh.flush()

        print(
            f"{epoch:>6}  {baseline:>8.4f}  {e_snd:>{col_w}.4f}  "
            f"{e_rcv:>{col_w}.4f}  {e_acc:>{col_w}.4f}  "
            f"{val_loss:>{col_w}.4f}  {val_acc:>{col_w}.4f}"
        )

        # -- Early stopping ----------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            best_sender_state   = {k: v.cpu().clone() for k, v in sender.state_dict().items()}
            best_receiver_state = {k: v.cpu().clone() for k, v in receiver.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n{tag} Early stopping triggered after {patience} epochs without improvement.")
                break

    elapsed = time.time() - t0
    csv_fh.close()

    if best_sender_state:
        sender.load_state_dict(best_sender_state)
        receiver.load_state_dict(best_receiver_state)
        print(f"{tag} Restored best model weights from Epoch {best_epoch} (val_loss: {best_val_loss:.4f})")

    print(f"{tag} Total gradient steps taken: {total_steps}")
    print()
    print(f"{tag} Finished in {elapsed:.1f}s.  Log: {log_path}")


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    train(cfg)


if __name__ == "__main__":
    import argparse
    _default_cfg = str(_ROOT / "configs" / "default.yaml")
    parser = argparse.ArgumentParser(description="Train emergent communication via REINFORCE")
    parser.add_argument("--config", type=str, default=_default_cfg)
    args = parser.parse_args()
    main(args.config)
