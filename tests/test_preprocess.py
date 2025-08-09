# ./tests/test_preprocess.py
# version: 2025.08.03.133000
# This file contains automated tests for the data preprocessing script.

import os
import sys
import json
import pytest
import tempfile
import shutil
import yaml

# Add the project root to the Python path to allow package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zo.sia.preprocess import build_full_index

@pytest.fixture
def mock_fs_for_preprocess():
    """
    A fixture to create a temporary file system with mock data sources
    for testing the preprocessing script.
    """
    tmp_dir = tempfile.mkdtemp()
    
    # --- Create Mock File System Structure ---
    mock_corpus_dir = os.path.join(tmp_dir, 'corpus')
    mock_train_dir = os.path.join(mock_corpus_dir, 'train')
    mock_template_dir = os.path.join(tmp_dir, 'templates')
    mock_processed_dir = os.path.join(tmp_dir, 'processed')
    
    os.makedirs(mock_train_dir)
    os.makedirs(mock_template_dir)
    os.makedirs(mock_processed_dir)

    # --- Create Mock Data Files ---
    mock_tsv_path = os.path.join(mock_train_dir, 'sample.tsv')
    with open(mock_tsv_path, 'w', encoding='utf-8') as f:
        f.write("en\tzo\nhello world\tleitung chibai\ngood morning\tzingtho nuam\n")
        
    mock_datasets_index_path = os.path.join(mock_corpus_dir, 'datasets.yaml')
    with open(mock_datasets_index_path, 'w', encoding='utf-8') as f:
        yaml.dump({'train_dir': mock_train_dir, 'en-zo': {'train': ['sample']}}, f)
        
    mock_template_path = os.path.join(mock_template_dir, 'sample.yaml')
    with open(mock_template_path, 'w', encoding='utf-8') as f:
        yaml.dump({'templates': [{'en': 'test template', 'zo': 'template siam'}]}, f)

    # --- Create a Mock Config Object ---
    class MockConfig:
        CORPUS_DIR = mock_corpus_dir
        TEMPLATE_DIR = mock_template_dir
        SHARED_TEMPLATE_DIR = "" # Not needed for this test
        PROCESSED_DATA_DIR = mock_processed_dir
        TMP_DIR = tmp_dir
        DATASETS_INDEX_FILE = mock_datasets_index_path

    yield MockConfig()
    
    # --- Teardown ---
    shutil.rmtree(tmp_dir)


def test_preprocess_creates_correct_index(mock_fs_for_preprocess):
    """
    Tests that the preprocess script runs and creates a data_index.json
    with the correct number of entries from both TSV and template files.
    """
    # Run the core logic of the preprocessor with the mock config
    build_full_index('en', 'zo', mock_fs_for_preprocess)

    # Verify the output
    index_path = os.path.join(mock_fs_for_preprocess.PROCESSED_DATA_DIR, "data_index.json")
    assert os.path.exists(index_path)

    with open(index_path, 'r') as f:
        data_index = json.load(f)
    
    # Should contain 2 entries from the TSV and 1 from the template
    assert len(data_index['train']) == 3
    assert len(data_index['test']) == 0
    
    # Check that one of the entries points to the generated template file
    generated_tsv_path = os.path.join(mock_fs_for_preprocess.TMP_DIR, "generated_template_data.tsv")
    assert any(entry[0] == generated_tsv_path for entry in data_index['train'])
