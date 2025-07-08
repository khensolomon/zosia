# tests/test_models.py
#
# What it does:
# This file contains unit tests for the neural network models defined in
# `zo/sia/model.py`. It verifies that the models can be instantiated and that
# the tensor shapes flow correctly through the layers for both single-item
# evaluation and batched training.
#
# How to run it:
# Run pytest from the project's root directory:
#   pytest tests/test_models.py

import pytest
import torch
from zo.sia.model import EncoderRNN, AttnDecoderRNN

# --- Configuration for the test models ---
VOCAB_SIZE = 100
HIDDEN_SIZE = 32
MAX_LENGTH = 15
BATCH_SIZE = 4

@pytest.fixture
def models():
    """A pytest fixture to create instances of our models for testing."""
    encoder = EncoderRNN(VOCAB_SIZE, HIDDEN_SIZE)
    decoder = AttnDecoderRNN(HIDDEN_SIZE, VOCAB_SIZE, max_length=MAX_LENGTH)
    return encoder, decoder

def test_encoder_forward_pass(models):
    """Tests the forward pass of the EncoderRNN for both batch and single item."""
    encoder, _ = models
    
    # Test with a batch of data
    input_batch = torch.randint(0, VOCAB_SIZE, (MAX_LENGTH, BATCH_SIZE))
    hidden_batch = encoder.initHidden(device='cpu', batch_size=BATCH_SIZE)
    output_batch, hidden_out_batch = encoder(input_batch, hidden_batch)
    
    assert output_batch.shape == (MAX_LENGTH, BATCH_SIZE, HIDDEN_SIZE)
    assert hidden_out_batch.shape == (1, BATCH_SIZE, HIDDEN_SIZE)

    # Test with a single item (like in evaluation)
    input_single = torch.randint(0, VOCAB_SIZE, (MAX_LENGTH, 1))
    hidden_single = encoder.initHidden(device='cpu', batch_size=1)
    output_single, hidden_out_single = encoder(input_single, hidden_single)
    
    assert output_single.shape == (MAX_LENGTH, 1, HIDDEN_SIZE)
    assert hidden_out_single.shape == (1, 1, HIDDEN_SIZE)

def test_decoder_forward_pass(models):
    """Tests the forward pass of the AttnDecoderRNN."""
    encoder, decoder = models
    
    # --- Setup dummy data for the decoder ---
    # The decoder needs an input token, a hidden state, and encoder outputs.
    
    # 1. Create dummy encoder outputs
    encoder_outputs = torch.randn(MAX_LENGTH, BATCH_SIZE, HIDDEN_SIZE)
    
    # 2. Create dummy decoder input token and hidden state
    decoder_input = torch.randint(0, VOCAB_SIZE, (1, BATCH_SIZE))
    decoder_hidden = torch.randn(1, BATCH_SIZE, HIDDEN_SIZE)

    # --- Perform the forward pass ---
    output, hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)

    # --- Assert output shapes ---
    assert output.shape == (BATCH_SIZE, VOCAB_SIZE)
    assert hidden.shape == (1, BATCH_SIZE, HIDDEN_SIZE)
    assert attn_weights.shape == (BATCH_SIZE, MAX_LENGTH)
