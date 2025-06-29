# -----------------------------------------------------------------------------
# File: tests/unit/test_models.py
#
# Description:
#   This file contains unit tests for the Transformer model components.
#   It verifies that the model can be built and can perform a forward pass
#   with the correct tensor shapes.
#
# Usage:
#   Run tests from the root directory of the project using pytest:
#   `pytest tests/unit`
# -----------------------------------------------------------------------------

import torch
import pytest
from src.models.transformer import build_transformer

@pytest.fixture
def model_config():
    """Provides a standard configuration for a test model."""
    return {
        "src_vocab_size": 100,
        "tgt_vocab_size": 110,
        "src_pad_idx": 0,
        "tgt_pad_idx": 0,
        "d_model": 64,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "num_heads": 4,
        "d_ff": 128,
        "dropout": 0.1,
        "max_seq_len": 50
    }

def test_build_transformer(model_config):
    """
    Tests if the build_transformer factory function can create a model
    without raising an error.
    """
    try:
        model = build_transformer(**model_config)
        assert model is not None, "Model should not be None"
    except Exception as e:
        pytest.fail(f"build_transformer failed with an exception: {e}")

def test_transformer_forward_pass(model_config):
    """
    Tests if the full model can perform a forward pass and produce an
    output tensor with the expected shape.
    """
    model = build_transformer(**model_config)
    
    # Create dummy input tensors
    batch_size = 10
    src_len = 15
    tgt_len = 18
    
    # Input tensors should have values within the vocabulary size
    src = torch.randint(1, model_config["src_vocab_size"], (batch_size, src_len))
    tgt = torch.randint(1, model_config["tgt_vocab_size"], (batch_size, tgt_len))
    
    # Create dummy masks (all True for this simple test)
    src_mask = torch.ones(batch_size, 1, 1, src_len, dtype=torch.bool)
    tgt_mask = torch.ones(batch_size, 1, tgt_len, tgt_len, dtype=torch.bool)
    
    try:
        # Perform forward pass
        encoder_output = model.encode(src, src_mask)
        decoder_output = model.decode(encoder_output, src_mask, tgt, tgt_mask)
        output = model.project(decoder_output)
        
        # Check output shape
        expected_shape = (batch_size, tgt_len, model_config["tgt_vocab_size"])
        assert output.shape == expected_shape, \
            f"Model output shape is incorrect. Expected {expected_shape}, but got {output.shape}"
            
    except Exception as e:
        pytest.fail(f"Model forward pass failed with an exception: {e}")

