# ./tests/test_pipeline.py
# version: 2025.08.03.120000
# This file contains automated tests for the main NMT pipeline.

import os
import sys
import json
import torch
import pytest
import tempfile
import shutil
import argparse

# Add the project root to the Python path to allow package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zo.sia.main import (
    Seq2SeqTransformer,
    Config,
    create_mask,
    main as main_py_main
)

@pytest.fixture
def mock_env_for_pipeline():
    """
    A fixture to create a temporary, isolated environment for testing
    the main training and inference pipeline. It creates a mock data index
    and tokenizer file.
    """
    tmp_dir = tempfile.mkdtemp()
    
    # --- Create Mock File System Structure ---
    mock_processed_dir = os.path.join(tmp_dir, 'processed')
    mock_experiments_dir = os.path.join(tmp_dir, 'experiments')
    os.makedirs(mock_processed_dir)
    os.makedirs(mock_experiments_dir)

    # --- Create Mock Data Index ---
    mock_data_file = os.path.join(tmp_dir, 'mock_data.tsv')
    with open(mock_data_file, 'w', encoding='utf-8') as f:
        f.write("en\tzo\nhello world\tleitung chibai\n")
    
    mock_index = {'train': [(mock_data_file, 2)], 'test': [(mock_data_file, 2)]}
    mock_index_path = os.path.join(mock_processed_dir, 'data_index.json')
    with open(mock_index_path, 'w') as f:
        json.dump(mock_index, f)

    # --- Create a dummy tokenizer model ---
    # In a real scenario, this would be a trained SentencePiece model.
    # For this test, an empty file is sufficient to pass the existence check.
    tokenizer_path = os.path.join(mock_experiments_dir, 'tokenizer_en-zo.model')
    with open(tokenizer_path, 'w') as f:
        f.write("") # Create empty file

    # --- Patch Config Paths ---
    original_paths = {
        'PROCESSED_DATA_DIR': Config.PROCESSED_DATA_DIR,
        'EXPERIMENTS_DIR': Config.EXPERIMENTS_DIR,
        'TMP_DIR': Config.TMP_DIR
    }
    
    Config.PROCESSED_DATA_DIR = mock_processed_dir
    Config.EXPERIMENTS_DIR = mock_experiments_dir
    Config.TMP_DIR = tmp_dir
    
    yield
    
    # --- Teardown ---
    for attr, path in original_paths.items():
        setattr(Config, attr, path)
    shutil.rmtree(tmp_dir)


def test_model_creation_and_forward_pass():
    """
    A "smoke test" to ensure the Transformer model can be created and can
    process a batch of dummy data without crashing.
    """
    model = Seq2SeqTransformer(num_encoder_layers=2, num_decoder_layers=2, embed_size=64, nhead=4, vocab_size=100, ff_hidden_size=128)
    src = torch.randint(0, 100, (4, 10))
    tgt = torch.randint(0, 100, (4, 10))
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt, 'cpu')
    output = model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
    assert output.shape == (4, 10, 100)

# We can add a more comprehensive end-to-end test here later if needed.
# For now, the model creation test serves as a good basic check.

