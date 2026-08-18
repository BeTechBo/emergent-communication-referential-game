"""
src/train_gumbel.py
-----------------------------------------------------------------------------
Phase 3 - Gumbel-Softmax training for the emergent-communication referential
game (Havrylov & Titov, NeurIPS 2017).

What this module adds over Phase 2
------------------------------------
Phase 2 tested Sender and Receiver in isolation, with the Sender using a
zero-feedback forward pass.  Phase 3 activates the full autoregressive
training loop:

  1. gumbel_generate() replaces Sender.forward() with a proper step-by-step
     generation loop.  Each timestep feeds its Gumbel-Softmax sample back as
     the next timestep input via Sender.symbol_embedding, giving the Sender
     real sequential information during training.

  2. The Gumbel-Softmax straight-through estimator (hard=True) keeps the
     entire pipeline differentiable: gradients flow from the Receiver's loss
     back through the Sender's LSTM and into concept_encoder.

  3. The training loop logs loss and accuracy to a CSV file each epoch.

Gradient flow path (Gumbel-Softmax)
-------------------------------------
  loss
    --> receiver scores        (cross-entropy)
    --> receiver.symbol_proj   (soft message input)
    --> soft_msg[:,t,:]        (gumbel_softmax, straight-through)
    --> sender logits at t     (Sender.step output_proj)
    --> sender LSTM state h_t  (accumulated from h_0)
    --> sender h_0, c_0        (Sender.encode)
    --> sender.concept_encoder (MLP on target concept vector)

Public API
----------
gumbel_generate(sender, concepts, temperature, hard) -> (B, L, vocab_size)
    Autoregressive Gumbel-Softmax message generation.

train_step(sender, receiver, batch, optimizer, temperature) -> (loss, acc)
    One forward + backward + optimizer step.

train(config) -> None
    Full training loop from a config dict.

load_config(path) -> dict
    Load a YAML config, merged with built-in defaults.

main(config_path) -> None
    CLI entry point.
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

# Allow running from any working directory.
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
    "gumbel": {
        "start_temperature": 1.0,
        "end_temperature": 1.0,
        "decay_rate": 1.0,
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
        "log_file": "train_gumbel.csv",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None = None) -> Dict[str, Any]:
    """
    Load a YAML config file and merge with DEFAULT_CONFIG.

    Falls back gracefully if PyYAML is not installed or the file is missing.
    """
    if path is None:
        return DEFAULT_CONFIG

    try:
        import yaml  # type: ignore
        with open(path, "r") as fh:
            file_cfg = yaml.safe_load(fh) or {}
        return _deep_merge(DEFAULT_CONFIG, file_cfg)
    except ImportError:
        print(
            "[train_gumbel] Warning: PyYAML not installed - using default config. "
            "Install with: pip install pyyaml",
            file=sys.stderr,
        )
        return DEFAULT_CONFIG
    except FileNotFoundError:
        print(
            f"[train_gumbel] Warning: config file '{path}' not found - "
            "using default config.",
            file=sys.stderr,
        )
        return DEFAULT_CONFIG


# -----------------------------------------------------------------------------
# Core: Gumbel-Softmax autoregressive generation
# -----------------------------------------------------------------------------

def gumbel_generate(
    sender: Sender,
    concepts: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = True,
) -> torch.Tensor:
    """
    Generate a message from the Sender using the Gumbel-Softmax
    straight-through estimator, with full autoregressive feedback.

    This is the Phase-3 replacement for Sender.forward() (which used a
    zero-feedback pass).  The autoregressive loop lets the Sender condition
    each symbol on all previously emitted symbols, and keeps the entire
    pipeline end-to-end differentiable.

    Step-by-step at timestep t
    ---------------------------
    1. sender.step(inp, h, c) -> logits (B, vocab_size)
    2. soft_sym = gumbel_softmax(logits, tau=temperature, hard=hard)
       - hard=True (default): returns a one-hot in the forward pass but
         routes gradients through the soft Gumbel distribution in the
         backward pass (straight-through estimator).
       - hard=False: returns a soft distribution (used in tests and eval).
    3. inp_next = soft_sym @ sender.symbol_embedding.weight
       Maps (B, vocab_size) x (vocab_size, embed_dim) -> (B, embed_dim).
       Equivalent to an embedding lookup when soft_sym is one-hot, but
       differentiable for soft inputs (gradients flow back through the
       Gumbel sampling to the logits).

    Parameters
    ----------
    sender   : Sender (Phase-2 module, unchanged)
    concepts : (B, concept_dim) float32  -- target concept vectors
    temperature : float  -- Gumbel-Softmax temperature
    hard     : bool  -- True = straight-through one-hot; False = soft dist

    Returns
    -------
    soft_msg : (B, max_length, vocab_size) float32
        Full message as a sequence of Gumbel-Softmax symbol distributions.
        Fed directly into Receiver.forward() via its soft input path.
    """
    B = concepts.size(0)
    h, c = sender.encode(concepts)       # (B, hidden_dim) each
    inp  = sender.sos_input(B)           # (B, embed_dim) -- learnable SOS

    soft_syms: List[torch.Tensor] = []
    for _ in range(sender.max_length):
        logits, h, c = sender.step(inp, h, c)           # (B, vocab_size)

        # Straight-through Gumbel-Softmax:
        #   forward  -- hard one-hot (argmax of (logits + Gumbel noise) / tau)
        #   backward -- gradient of the soft Gumbel distribution w.r.t. logits
        soft_sym = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        soft_syms.append(soft_sym)                       # (B, vocab_size)

        # Autoregressive feedback: embed the (soft) symbol for next timestep.
        # soft_sym @ W maps (B, V) x (V, E) -> (B, E).
        # Gradients flow through this matrix multiply back to soft_sym,
        # then via the straight-through to logits, and finally to the LSTM.
        inp = soft_sym @ sender.symbol_embedding.weight  # (B, embed_dim)

    return torch.stack(soft_syms, dim=1)                 # (B, L, V)


# -----------------------------------------------------------------------------
# One training step
# -----------------------------------------------------------------------------

def train_step(
    sender: Sender,
    receiver: Receiver,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    temperature: float = 1.0,
) -> Tuple[float, float]:
    """
    One forward + backward + optimizer step.

    Parameters
    ----------
    sender, receiver : Phase-2 agents (modified in-place by optimizer)
    batch : dict from sample_batch() -- requires keys:
        "target_vectors"   (B, concept_dim) float32
        "all_candidates"   (B, N+1, concept_dim) float32
        "target_indices"   (B,) int64
    optimizer : Adam over sender + receiver parameters
    temperature : Gumbel-Softmax temperature

    Returns
    -------
    loss     : float  -- cross-entropy loss value
    accuracy : float  -- fraction where Receiver argmax == target_index
    """
    targets        = batch["target_vectors"]     # (B, concept_dim)
    candidates     = batch["all_candidates"]     # (B, N+1, concept_dim)
    target_indices = batch["target_indices"]     # (B,) int64

    optimizer.zero_grad()

    # 1. Sender: autoregressive Gumbel-Softmax generation
    soft_msg = gumbel_generate(sender, targets, temperature=temperature)
    # soft_msg: (B, max_length, vocab_size)

    # 2. Receiver: score each candidate given the message
    scores = receiver(soft_msg, candidates)      # (B, N+1)

    # 3. Cross-entropy loss against the true target position
    loss = F.cross_entropy(scores, target_indices)

    # 4. Backward pass + gradient step
    loss.backward()
    optimizer.step()

    # 5. Task accuracy
    predicted = scores.argmax(dim=-1)            # (B,)
    accuracy  = (predicted == target_indices).float().mean().item()

    return loss.item(), accuracy


class ConceptSubspace(ConceptSpace):
    """A subset of a ConceptSpace for train/val splits."""
    def __init__(self, vocab_sizes: List[int], subset: List[Tuple[int, ...]]) -> None:
        self.vocab_sizes = tuple(vocab_sizes)
        self.n_attributes = len(vocab_sizes)
        self.vector_dim = sum(vocab_sizes)
        self._all_concepts = subset
        self.n_concepts = len(subset)


# -----------------------------------------------------------------------------
# Full training loop
# -----------------------------------------------------------------------------

def train(config: Dict[str, Any]) -> None:
    """
    Full Gumbel-Softmax training loop.

    Reads all hyperparameters from `config` (use load_config() to produce one).
    Logs epoch, loss, and accuracy to a CSV file specified in config["logging"].
    """
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
    
    rng  = random.Random(seed)

    # -- Hyperparameters -------------------------------------------------------
    epochs        = config["training"]["epochs"]
    batch_size    = config["training"]["batch_size"]
    lr            = config["training"]["learning_rate"]
    weight_decay  = config["training"]["weight_decay"]
    patience      = config["training"]["patience"]

    start_temp    = config["gumbel"]["start_temperature"]
    end_temp      = config["gumbel"]["end_temperature"]
    decay_rate    = config["gumbel"]["decay_rate"]

    vocab_sizes   = list(config["data"]["vocab_sizes"])
    n_distractors = config["data"]["n_distractors"]
    concept_dim   = sum(vocab_sizes)

    vocab_size    = config["model"]["vocab_size"]
    embed_dim     = config["model"]["embed_dim"]
    hidden_dim    = config["model"]["hidden_dim"]
    max_length    = config["model"]["max_length"]
    mlp_hidden    = config["model"]["mlp_hidden_dim"]

    log_dir  = config["logging"]["log_dir"]
    log_file = config["logging"]["log_file"]

    # -- Concept space & Train/Val split ---------------------------------------
    cs = ConceptSpace(vocab_sizes)
    all_concepts = cs.all_concepts
    rng.shuffle(all_concepts)
    
    split_idx = int(len(all_concepts) * 0.8)
    train_concepts = all_concepts[:split_idx]
    val_concepts = all_concepts[split_idx:]
    
    train_cs = ConceptSubspace(vocab_sizes, train_concepts)
    val_cs = ConceptSubspace(vocab_sizes, val_concepts)

    # -- Header ----------------------------------------------------------------
    print(f"[train_gumbel] Concept space : {cs.n_concepts} total concepts")
    print(f"[train_gumbel] Train / Val   : {train_cs.n_concepts} / {val_cs.n_concepts}")
    print(f"[train_gumbel] Epochs        : {epochs} (patience={patience})")
    print(f"[train_gumbel] Batch size    : {batch_size}")
    print(f"[train_gumbel] LR            : {lr} (wd={weight_decay})")
    print(f"[train_gumbel] Temperature   : {start_temp} -> {end_temp} (decay={decay_rate})")
    print(f"[train_gumbel] Vocab sizes   : {vocab_sizes}  (concept_dim={concept_dim})")
    print(f"[train_gumbel] Msg vocab     : {vocab_size}  max_len={max_length}")
    print(f"[train_gumbel] Model config  : embed_dim={embed_dim}, hidden_dim={hidden_dim}, mlp_hidden_dim={mlp_hidden}")

    # -- Models ----------------------------------------------------------------
    sender   = Sender(concept_dim, vocab_size, embed_dim, hidden_dim,
                      max_length, mlp_hidden_dim=mlp_hidden)
    receiver = Receiver(vocab_size, embed_dim, hidden_dim, concept_dim)

    n_params = (
        sum(p.numel() for p in sender.parameters()) +
        sum(p.numel() for p in receiver.parameters())
    )
    print(f"[train_gumbel] Parameters    : {n_params:,}")

    # -- Optimizer -------------------------------------------------------------
    optimizer = torch.optim.Adam(
        list(sender.parameters()) + list(receiver.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # -- Logging setup ---------------------------------------------------------
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    csv_fh   = open(log_path, "w", newline="")
    writer   = csv.writer(csv_fh)
    writer.writerow(["epoch", "temperature", "train_loss", "train_acc", "val_loss", "val_acc"])
    print(f"[train_gumbel] Logging to    : {log_path}")
    print()

    # -- Training loop ---------------------------------------------------------
    col_w = 9
    header = f"{'Epoch':>6}  {'Temp':>5}  {'Trn Loss':>{col_w}}  {'Trn Acc':>{col_w}}  {'Val Loss':>{col_w}}  {'Val Acc':>{col_w}}"
    print(header)
    print("-" * len(header))

    best_val_loss = float("inf")
    patience_counter = 0
    best_sender_state = None
    best_receiver_state = None
    best_epoch = 0

    t0 = time.time()
    total_steps = 0
    for epoch in range(1, epochs + 1):
        sender.train()
        receiver.train()

        # Temperature annealing
        temperature = max(end_temp, start_temp * (decay_rate ** (epoch - 1)))

        # One epoch = iterating through the train set
        steps_per_epoch = max(1, train_cs.n_concepts // batch_size)
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        for _ in range(steps_per_epoch):
            train_batch = sample_batch(
                batch_size,
                vocab_sizes=vocab_sizes,
                n_distractors=n_distractors,
                rng=rng,
                concept_space=train_cs,
            )
            step_loss, step_acc = train_step(sender, receiver, train_batch, optimizer, temperature)
            epoch_loss += step_loss
            epoch_acc += step_acc
            total_steps += 1
            
        loss = epoch_loss / steps_per_epoch
        acc = epoch_acc / steps_per_epoch
        
        # Validation
        sender.eval()
        receiver.eval()
        with torch.no_grad():
            # Adjust batch size for validation if it's larger than val set
            val_batch_size = min(batch_size, val_cs.n_concepts)
            val_batch = sample_batch(
                val_batch_size,
                vocab_sizes=vocab_sizes,
                n_distractors=n_distractors,
                rng=rng,
                concept_space=val_cs,
            )
            # Evaluate using soft_msg without gradients
            val_soft_msg = gumbel_generate(sender, val_batch["target_vectors"], temperature=temperature, hard=False)
            val_scores = receiver(val_soft_msg, val_batch["all_candidates"])
            val_loss = F.cross_entropy(val_scores, val_batch["target_indices"]).item()
            val_predicted = val_scores.argmax(dim=-1)
            val_acc = (val_predicted == val_batch["target_indices"]).float().mean().item()

        writer.writerow([epoch, f"{temperature:.3f}", f"{loss:.6f}", f"{acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}"])
        csv_fh.flush()

        print(f"{epoch:>6}  {temperature:>5.2f}  {loss:>{col_w}.4f}  {acc:>{col_w}.4f}  {val_loss:>{col_w}.4f}  {val_acc:>{col_w}.4f}")
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            best_sender_state = {k: v.cpu().clone() for k, v in sender.state_dict().items()}
            best_receiver_state = {k: v.cpu().clone() for k, v in receiver.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[train_gumbel] Early stopping triggered! Validation loss hasn't improved for {patience} epochs.")
                break

    elapsed = time.time() - t0
    csv_fh.close()
    
    if best_sender_state is not None and best_receiver_state is not None:
        sender.load_state_dict(best_sender_state)
        receiver.load_state_dict(best_receiver_state)
        print(f"[train_gumbel] Restored best model weights from Epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
    
    print(f"[train_gumbel] Total gradient steps taken: {total_steps}")
    print()
    print(f"[train_gumbel] Finished in {elapsed:.1f}s.  Log: {log_path}")


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    train(cfg)


if __name__ == "__main__":
    import argparse
    _default_cfg = str(_ROOT / "configs" / "default.yaml")
    parser = argparse.ArgumentParser(
        description="Train emergent communication via Gumbel-Softmax"
    )
    parser.add_argument(
        "--config", type=str, default=_default_cfg,
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
