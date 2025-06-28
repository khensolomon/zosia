# unit tests for individual components of your Transformer model. We'll use dummy inputs to check shapes and basic functionality.

import pytest
import torch
import torch.nn as nn

# Corrected imports based on your transformer_model.py
from src.models.transformer_model import (
    Transformer,
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    PositionwiseFeedforwardLayer # Included for completeness, though it's also imported within layers
)
from src.models.attention import MultiHeadAttention # Still needed for MultiHeadAttention test
from src.models.embeddings import TokenEmbedding, PositionalEncoding # Still needed for embedding tests

# --- Fixtures for common test parameters ---
@pytest.fixture(scope="module")
def device():
    """Provides a torch device (CPU for testing simplicity)."""
    return torch.device('cpu')

@pytest.fixture
def model_config():
    """Provides a basic configuration dictionary for the Transformer model and its components."""
    # These parameters directly map to your Encoder/Decoder/Layer __init__ args
    return {
        'src_vocab_size': 100,  # Dummy vocab size for source
        'tgt_vocab_size': 100,  # Dummy vocab size for target
        'hid_dim': 128,         # Dimension of hidden states (d_model)
        'n_heads': 8,           # Number of attention heads
        'pf_dim': 512,          # Feed-forward dimension (d_ff)
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'dropout': 0.1,
        'max_seq_len': 50,      # Max sequence length for positional encoding
        'src_pad_idx': 0,       # Dummy padding index for source
        'trg_pad_idx': 0        # Dummy padding index for target
    }

@pytest.fixture
def dummy_input(model_config, device):
    """Provides dummy input tensors for testing."""
    batch_size = 2
    src_seq_len = 10
    trg_seq_len = 12 # Target sequence might be different length

    src_vocab_size = model_config['src_vocab_size']
    tgt_vocab_size = model_config['tgt_vocab_size']
    src_pad_idx = model_config['src_pad_idx']
    trg_pad_idx = model_config['trg_pad_idx']

    # Dummy token IDs, ensuring some padding tokens for mask testing
    src_tokens = torch.randint(1, src_vocab_size, (batch_size, src_seq_len), device=device)
    src_tokens[0, src_seq_len // 2:] = src_pad_idx # Add some padding
    tgt_tokens = torch.randint(1, tgt_vocab_size, (batch_size, trg_seq_len), device=device)
    tgt_tokens[1, trg_seq_len // 2:] = trg_pad_idx # Add some padding

    # Make masks using the Transformer's internal make_src_mask/make_trg_mask logic
    # (We'll use dummy Transformer instance's methods for this in main test)
    # For individual layer tests, we'll create simpler masks or rely on their internal masking.

    return src_tokens, tgt_tokens, src_pad_idx, trg_pad_idx, batch_size, src_seq_len, trg_seq_len

# --- Unit Tests for Individual Components ---

def test_token_embedding(model_config, device):
    """Test TokenEmbedding layer output shape."""
    vocab_size = model_config['src_vocab_size']
    hid_dim = model_config['hid_dim']
    embedding_layer = TokenEmbedding(vocab_size, hid_dim).to(device)

    # Dummy input: batch_size=2, seq_len=5
    input_tokens = torch.randint(0, vocab_size, (2, 5), device=device)
    output = embedding_layer(input_tokens)

    assert output.shape == (2, 5, hid_dim)
    assert output.dtype == torch.float32

def test_positional_encoding(model_config, device):
    """Test PositionalEncoding layer output shape and values (sanity check)."""
    hid_dim = model_config['hid_dim']
    max_seq_len = model_config['max_seq_len']
    dropout = model_config['dropout'] # PositionalEncoding also takes dropout and device
    pos_encoding = PositionalEncoding(hid_dim, max_seq_len, dropout, device).to(device)

    # Dummy input: batch_size=2, seq_len=10, hid_dim=128
    dummy_embedding = torch.randn(2, 10, hid_dim, device=device)
    output = pos_encoding(dummy_embedding)

    assert output.shape == dummy_embedding.shape
    # Check that positional encoding adds non-zero values
    # It adds a small value, so `allclose` might pass if values are too similar.
    # A simple check: sum of squared differences should not be zero.
    assert torch.sum((output - dummy_embedding)**2) > 1e-6

def test_multi_head_attention(model_config, device):
    """Test MultiHeadAttention layer output shape."""
    hid_dim = model_config['hid_dim']
    n_heads = model_config['n_heads']
    dropout = model_config['dropout']
    batch_size = 2
    seq_len = 10

    mha_layer = MultiHeadAttention(hid_dim, n_heads, dropout, device).to(device)

    # Dummy query, key, value tensors
    q = torch.randn(batch_size, seq_len, hid_dim, device=device)
    k = torch.randn(batch_size, seq_len, hid_dim, device=device)
    v = torch.randn(batch_size, seq_len, hid_dim, device=device)
    mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=device) # Dummy mask

    output, _ = mha_layer(q, k, v, mask)

    assert output.shape == (batch_size, seq_len, hid_dim)
    assert output.dtype == torch.float32

def test_positionwise_feedforward(model_config, device):
    """Test PositionwiseFeedforwardLayer output shape."""
    hid_dim = model_config['hid_dim']
    pf_dim = model_config['pf_dim']
    dropout = model_config['dropout']
    batch_size = 2
    seq_len = 10

    ffn_layer = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout).to(device)

    # Dummy input tensor
    x = torch.randn(batch_size, seq_len, hid_dim, device=device)
    output = ffn_layer(x)

    assert output.shape == (batch_size, seq_len, hid_dim)
    assert output.dtype == torch.float32

def test_encoder_layer(model_config, device):
    """Test EncoderLayer output shape."""
    hid_dim = model_config['hid_dim']
    n_heads = model_config['n_heads']
    pf_dim = model_config['pf_dim']
    dropout = model_config['dropout']
    batch_size = 2
    seq_len = 10

    encoder_layer = EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device).to(device)

    # Input to encoder layer is usually an embedding
    dummy_src_embed = torch.randn(batch_size, seq_len, hid_dim, device=device)
    # Mask shape: (batch_size, 1, 1, src_len)
    src_mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool, device=device)
    output = encoder_layer(dummy_src_embed, src_mask)

    assert output.shape == (batch_size, seq_len, hid_dim)
    assert output.dtype == torch.float32

def test_decoder_layer(model_config, device):
    """Test DecoderLayer output shape."""
    hid_dim = model_config['hid_dim']
    n_heads = model_config['n_heads']
    pf_dim = model_config['pf_dim']
    dropout = model_config['dropout']
    batch_size = 2
    seq_len_trg = 12
    seq_len_src = 10

    decoder_layer = DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device).to(device)

    # Input to decoder layer (target embedding) and encoder output
    dummy_trg_embed = torch.randn(batch_size, seq_len_trg, hid_dim, device=device)
    dummy_enc_output = torch.randn(batch_size, seq_len_src, hid_dim, device=device)

    # Mask shapes: trg_mask (batch_size, 1, trg_len, trg_len), src_mask (batch_size, 1, 1, src_len)
    trg_mask = torch.ones(batch_size, 1, seq_len_trg, seq_len_trg, dtype=torch.bool, device=device) # Simplified for test
    src_mask = torch.ones(batch_size, 1, 1, seq_len_src, dtype=torch.bool, device=device) # Simplified for test

    output, _, _ = decoder_layer(dummy_trg_embed, dummy_enc_output, trg_mask, src_mask)

    assert output.shape == (batch_size, seq_len_trg, hid_dim)
    assert output.dtype == torch.float32

def test_encoder(model_config, dummy_input, device):
    """Test full Encoder output shape."""
    src_tokens, _, src_pad_idx, _, batch_size, src_seq_len, _ = dummy_input

    encoder = Encoder(
        input_dim=model_config['src_vocab_size'],
        hid_dim=model_config['hid_dim'],
        n_layers=model_config['num_encoder_layers'],
        n_heads=model_config['n_heads'],
        pf_dim=model_config['pf_dim'],
        dropout=model_config['dropout'],
        max_seq_len=model_config['max_seq_len'],
        device=device
    ).to(device)

    # Make src mask (assuming Transformer's make_src_mask logic)
    src_mask = (src_tokens != src_pad_idx).unsqueeze(1).unsqueeze(2)

    output = encoder(src_tokens, src_mask)
    assert output.shape == (batch_size, src_seq_len, model_config['hid_dim'])
    assert output.dtype == torch.float32

def test_decoder(model_config, dummy_input, device):
    """Test full Decoder output shape."""
    _, tgt_tokens, _, trg_pad_idx, batch_size, _, trg_seq_len = dummy_input
    
    decoder = Decoder(
        output_dim=model_config['tgt_vocab_size'],
        hid_dim=model_config['hid_dim'],
        n_layers=model_config['num_decoder_layers'],
        n_heads=model_config['n_heads'],
        pf_dim=model_config['pf_dim'],
        dropout=model_config['dropout'],
        max_seq_len=model_config['max_seq_len'],
        device=device
    ).to(device)

    # Dummy encoder output
    dummy_enc_output = torch.randn(batch_size, 10, model_config['hid_dim'], device=device) # Assuming 10 src_len
    
    # Make masks (assuming Transformer's make_trg_mask and make_src_mask logic)
    src_mask = (torch.ones(batch_size, 10, dtype=torch.long, device=device) != 0).unsqueeze(1).unsqueeze(2) # Dummy src mask
    trg_pad_mask = (tgt_tokens != trg_pad_idx).unsqueeze(1).unsqueeze(2)
    trg_sub_mask = torch.tril(torch.ones((trg_seq_len, trg_seq_len), device=device)).bool()
    trg_mask = trg_pad_mask & trg_sub_mask

    output = decoder(tgt_tokens, dummy_enc_output, trg_mask, src_mask)
    assert output.shape == (batch_size, trg_seq_len, model_config['tgt_vocab_size']) # Final linear layer
    assert output.dtype == torch.float32


def test_transformer_model_forward_pass(model_config, dummy_input, device):
    """Test the full Transformer model's forward pass output shape."""
    src_tokens, tgt_tokens, src_pad_idx, trg_pad_idx, batch_size, src_seq_len, trg_seq_len = dummy_input

    # Instantiate Encoder and Decoder first
    encoder = Encoder(
        input_dim=model_config['src_vocab_size'],
        hid_dim=model_config['hid_dim'],
        n_layers=model_config['num_encoder_layers'],
        n_heads=model_config['n_heads'],
        pf_dim=model_config['pf_dim'],
        dropout=model_config['dropout'],
        max_seq_len=model_config['max_seq_len'],
        device=device
    ).to(device)

    decoder = Decoder(
        output_dim=model_config['tgt_vocab_size'],
        hid_dim=model_config['hid_dim'],
        n_layers=model_config['num_decoder_layers'],
        n_heads=model_config['n_heads'],
        pf_dim=model_config['pf_dim'],
        dropout=model_config['dropout'],
        max_seq_len=model_config['max_seq_len'],
        device=device
    ).to(device)

    # Instantiate the Transformer with Encoder and Decoder instances
    model = Transformer(encoder, decoder, src_pad_idx, trg_pad_idx, device).to(device)

    # Put model in evaluation mode for consistent dropout behavior
    model.eval()

    output = model(src_tokens, tgt_tokens)

    # The output should be raw logits for the target vocabulary
    assert output.shape == (batch_size, trg_seq_len, model_config['tgt_vocab_size'])
    assert output.dtype == torch.float32

    # Basic check for non-zero output (model isn't returning all zeros)
    assert torch.any(output != 0)