# ./tests/test_pipeline.py
# version: 2025.08.01.151000
# This file contains automated tests for our NMT script.
# To run these tests, navigate to the project root and run: pytest

import os
import sys
import argparse
import yaml
import torch
import pytest

# Add the script's directory to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

# Now we can import the functions and classes from our nmt script
from nmt import (
    get_indexed_pairs,
    generate_from_template_file,
    Seq2SeqTransformer,
    Tokenizer,
    Config,
    create_mask
)

# --- Test Fixtures and Setup ---

@pytest.fixture
def mock_env():
    """
    A pytest fixture to set up a mock environment for testing.
    It patches the Config class to point to mock data directories and
    handles cleanup automatically after each test.
    """
    # Define the paths to our mock data for testing
    MOCK_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../mock'))
    MOCK_CORPUS_DIR = os.path.join(MOCK_DATA_DIR, 'corpus')
    MOCK_TEMPLATE_DIR = os.path.join(MOCK_DATA_DIR, 'templates')
    MOCK_DATASETS_INDEX = os.path.join(MOCK_CORPUS_DIR, 'datasets.yaml')

    # Store original paths to restore them later
    original_index_path = Config.DATASETS_INDEX_FILE
    original_template_dir = Config.TEMPLATE_DIR

    # Patch the Config class with mock paths
    Config.DATASETS_INDEX_FILE = MOCK_DATASETS_INDEX
    Config.TEMPLATE_DIR = MOCK_TEMPLATE_DIR
    
    # Yield control to the test function, passing useful paths
    yield {
        "template_dir": MOCK_TEMPLATE_DIR
    }
    
    # Teardown: This code runs after the test function completes
    Config.DATASETS_INDEX_FILE = original_index_path
    Config.TEMPLATE_DIR = original_template_dir


# --- Test Functions ---

def test_get_indexed_pairs_loads_correctly(mock_env):
    """
    Tests if the data loader correctly parses a well-formed TSV file
    based on the mock datasets.yaml index.
    """
    pairs = get_indexed_pairs('en', 'zo', 'train')
    
    # Assert that it found the correct number of pairs
    assert len(pairs) == 2
    # Assert that the content is correct
    assert pairs[0] == ("hello world", "leitung chibai")
    assert pairs[1] == ("good morning", "zingtho nuam")

def test_template_engine_handles_conditionals(mock_env):
    """
    Tests if the advanced template engine correctly applies conditional logic
    based on metadata tags. This is a critical test for our most complex feature.
    """
    template_file = os.path.join(mock_env["template_dir"], 'sample_conditional_template.yaml')
    
    # Test the zo -> en direction
    pairs_zo_en = generate_from_template_file(template_file, 'zo', 'en')
    
    # Convert to a dictionary for easy lookup and assertion
    results_zo_en = {src: tgt for src, tgt in pairs_zo_en}

    print(results_zo_en)
    
    # Assert that the correct, capitalized sentences were generated
    assert results_zo_en["Pasalpa in tui a dawn nuam"] == "The man like to drink water"
    assert results_zo_en["Numeinu in uipa a kimawl pih nuam"] == "The woman like to play with the dog"

    # Test the en -> zo direction
    pairs_en_zo = generate_from_template_file(template_file, 'en', 'zo')
    results_en_zo = {src: tgt for src, tgt in pairs_en_zo}

    assert results_en_zo["The man like to drink water"] == "Pasalpa in tui a dawn nuam"
    assert results_en_zo["The woman like to play with the dog"] == "Numeinu in uipa a kimawl pih nuam"

def test_model_creation_and_forward_pass():
    """
    A "smoke test" to ensure the Transformer model can be created and can
    process a batch of dummy data without crashing.
    """
    # Create a dummy tokenizer and some dummy data
    vocab_size = 100
    batch_size = 4
    seq_len = 10
    
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Build the model
    model = Seq2SeqTransformer(
        num_encoder_layers=2,
        num_decoder_layers=2,
        embed_size=64,
        nhead=4,
        vocab_size=vocab_size,
        ff_hidden_size=128
    )

    # Perform a forward pass
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt, 'cpu')
    output = model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

    # Assert that the output shape is correct
    assert output.shape == (batch_size, seq_len, vocab_size)
