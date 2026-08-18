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
