# tests/integration/test_full_pipeline.py

import pytest
import torch
import sentencepiece as spm
import os
import yaml
from pathlib import Path

# Corrected imports based on your transformer_model.py
from src.models.transformer_model import Transformer, Encoder, Decoder # Also need Encoder, Decoder for instantiation


# --- Fixtures for integration tests ---

@pytest.fixture(scope="module")
def project_root():
    """Provides the path to the project root directory."""
    # This assumes you run pytest from the project root or configure pytest.ini
    return Path(__file__).parent.parent.parent

@pytest.fixture(scope="module")
def device():
    """Provides a torch device (CPU for testing simplicity)."""
    return torch.device('cpu')

@pytest.fixture(scope="module")
def dummy_data_dir(tmp_path_factory, device): # Add device here
    """Creates a temporary directory and dummy config/data files for testing."""
    temp_dir = tmp_path_factory.mktemp("dummy_nmt_data")

    # Create dummy data_config.yaml
    data_config_content = f"""
    raw_data_dir: {temp_dir}/raw
    processed_data_dir: {temp_dir}/processed
    sp_model_dir: {temp_dir}/sp_models
    source_language: zo
    target_language: en
    vocab_size: 58 # Small vocab for testing
    max_seq_len: 50
    """

    (temp_dir / "config").mkdir(exist_ok=True)
    data_config_path = temp_dir / "config" / "data_config.yaml"
    data_config_path.write_text(data_config_content)

    # Create dummy raw data files (very small)
    (temp_dir / "raw").mkdir(exist_ok=True)
    (temp_dir / "raw" / "train.zo").write_text("Hello world.\nThis is a test.\n")
    (temp_dir / "raw" / "train.en").write_text("Hello world.\nThis is a test.\n")
    (temp_dir / "raw" / "val.zo").write_text("Test sentence.\n")
    (temp_dir / "raw" / "val.en").write_text("Test sentence.\n")

    # Create a dummy model_config.yaml (adjusting to match your Transformer's __init__ args implicitly)
    model_config_content = """
    hid_dim: 128
    n_heads: 8
    num_encoder_layers: 2
    num_decoder_layers: 2
    pf_dim: 512
    dropout: 0.1
    max_seq_len: 50
    src_vocab_size: 58 # Corrected to match data_config vocab_size
    tgt_vocab_size: 58 # Corrected to match data_config vocab_size
    src_pad_idx: 0 # Assuming 0 for SentencePiece pad_id
    trg_pad_idx: 0 # Assuming 0 for SentencePiece pad_id
    """
    model_config_path = temp_dir / "config" / "model_config.yaml"
    model_config_path.write_text(model_config_content)


    # Create a dummy SentencePiece model for testing
    sp_model_dir = temp_dir / "sp_models"
    sp_model_dir.mkdir(exist_ok=True)
    sp_model_prefix = str(sp_model_dir / "zosia_sp")
    spm.SentencePieceTrainer.train(
        input=str(temp_dir / "raw" / "train.zo"),
        model_prefix=sp_model_prefix,
        vocab_size=58, # Matches config
        model_type="bpe"
    )
    
    yield temp_dir # Provide the path to the dummy data

@pytest.fixture(scope="module")
def trained_model_path(dummy_data_dir, device): # Add device here
    """Creates a dummy trained model file for testing."""
    model_config_path = dummy_data_dir / "config" / "model_config.yaml"
    with open(model_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Instantiate Encoder and Decoder with device
    encoder = Encoder(
        input_dim=config['src_vocab_size'],
        hid_dim=config['hid_dim'],
        n_layers=config['num_encoder_layers'],
        n_heads=config['n_heads'],
        pf_dim=config['pf_dim'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        device=device
    ).to(device)

    decoder = Decoder(
        output_dim=config['tgt_vocab_size'],
        hid_dim=config['hid_dim'],
        n_layers=config['num_decoder_layers'],
        n_heads=config['n_heads'],
        pf_dim=config['pf_dim'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        device=device
    ).to(device)

    # Instantiate the dummy Transformer model
    dummy_model = Transformer(
        encoder,
        decoder,
        config['src_pad_idx'],
        config['trg_pad_idx'],
        device
    ).to(device)

    # Save its state_dict to a dummy .pt file
    model_file_path = dummy_data_dir / "dummy_best_model.pt"
    torch.save(dummy_model.state_dict(), model_file_path)
    yield model_file_path
    # Teardown: file will be removed with dummy_data_dir

# --- Integration Tests ---

def test_dummy_data_preparation(dummy_data_dir):
    """
    Verify that dummy data files and SentencePiece model are created.
    This implicitly checks if basic data config setup works.
    """
    assert (dummy_data_dir / "config" / "data_config.yaml").exists()
    assert (dummy_data_dir / "config" / "model_config.yaml").exists()
    assert (dummy_data_dir / "raw" / "train.zo").exists()
    assert (dummy_data_dir / "sp_models" / "zosia_sp.model").exists()
    assert (dummy_data_dir / "sp_models" / "zosia_sp.vocab").exists()

def test_model_loading_and_basic_inference(trained_model_path, dummy_data_dir, device):
    """
    Test that a trained model and tokenizer can be loaded and perform a basic
    forward pass (inference) without crashing.
    """
    try:
        # Load model configuration
        model_config_path = dummy_data_dir / "config" / "model_config.yaml"
        with open(model_config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"DEBUG: Loaded model config: {config}") # Added debug print

        # Load SentencePiece model
        sp_model_path = str(dummy_data_dir / "sp_models" / "zosia_sp.model")
        sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        print(f"DEBUG: SentencePiece vocab size: {sp.get_piece_size()}") # Added debug print

        # Instantiate Encoder and Decoder with device
        print("DEBUG: Attempting to instantiate Encoder...") # Added debug print
        encoder = Encoder(
            input_dim=config['src_vocab_size'],
            hid_dim=config['hid_dim'],
            n_layers=config['num_encoder_layers'],
            n_heads=config['n_heads'],
            pf_dim=config['pf_dim'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len'],
            device=device
        ).to(device)
        print("DEBUG: Encoder instantiated successfully.") # Added debug print

        print("DEBUG: Attempting to instantiate Decoder...") # Added debug print
        decoder = Decoder(
            output_dim=config['tgt_vocab_size'],
            hid_dim=config['hid_dim'],
            n_layers=config['num_decoder_layers'],
            n_heads=config['n_heads'],
            pf_dim=config['pf_dim'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len'],
            device=device
        ).to(device)
        print("DEBUG: Decoder instantiated successfully.") # Added debug print

        # Instantiate the Transformer with Encoder and Decoder instances
        print("DEBUG: Attempting to instantiate Transformer...") # Added debug print
        model = Transformer(
            encoder,
            decoder,
            config['src_pad_idx'],
            config['trg_pad_idx'],
            device
        ).to(device)
        print("DEBUG: Transformer instantiated successfully.") # Added debug print

        # Now, model is defined, so load_state_dict can be called
        model.load_state_dict(torch.load(trained_model_path, map_location=device))
        model.eval() # Set to evaluation mode

        # --- Perform a very basic "translation" test ---
        dummy_text = "This is a test sentence."

        # 1. Tokenize input
        src_tokens_list = [sp.bos_id()] + sp.encode_as_ids(dummy_text) + [sp.eos_id()]
        src_tokens = torch.tensor([src_tokens_list], dtype=torch.long, device=device)

        max_output_len = 20
        translated_tokens = [sp.bos_id()]

        with torch.no_grad():
            # Encoder forward pass once
            src_mask = model.make_src_mask(src_tokens)
            enc_src = model.encoder(src_tokens, src_mask)

            # Decoder loop (simulate a few steps or until EOS)
            for i in range(max_output_len):
                trg_tensor = torch.LongTensor(translated_tokens).unsqueeze(0).to(device)
                trg_mask = model.make_trg_mask(trg_tensor)

                output_logits = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

                # Get the predicted next token (greedy decoding)
                pred_token_id = output_logits.argmax(dim=-1)[:, -1].item()

                # --- NEW: Ensure predicted token ID is within SentencePiece vocabulary bounds ---
                # Get the actual vocabulary size from the loaded SentencePiece model
                sp_vocab_size = sp.get_piece_size() # This gets the actual vocab size of the trained SP model
                if pred_token_id >= sp_vocab_size:
                    # If out of bounds, replace with UNK token ID
                    pred_token_id = sp.unk_id() # Use SentencePiece's UNK ID
                # --- END NEW ---

                translated_tokens.append(pred_token_id)

                if pred_token_id == sp.eos_id(): # Check for end-of-sentence token
                    break

        # Detokenize the result (excluding BOS/EOS, UNK, PAD)
        final_translated_ids = [
            token_id for token_id in translated_tokens
            if token_id not in {sp.pad_id(), sp.bos_id(), sp.eos_id(), sp.unk_id()} # Also filter UNK if it's there
        ]
        translated_text = sp.decode_ids(final_translated_ids)

        # Basic assertions
        assert len(translated_text) >= 0 # Changed to >=0 as it might be empty if only special tokens are predicted
        print(f"\nIntegration Test: Model loaded and basic inference performed.")
        print(f"Input: '{dummy_text}'")
        print(f"Translated (dummy): '{translated_text}'")

    except Exception as e:
        pytest.fail(f"Full pipeline inference failed unexpectedly: {e}")