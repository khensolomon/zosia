# zo/sia/data_loader.py
#
# What it does:
# This module is responsible for loading all datasets specified in the project
# configuration. It distinguishes between training and testing sets based on
# the `index.yaml` files in each data source directory.
#
# How to use it:
# Import the `load_all_data` function and pass it the project config object.
# It returns a dictionary with 'train' and 'test' keys.

import os
import yaml

def _load_pairs_from_files(source_path, pair_files_list):
    """Helper function to load sentence pairs from a list of files."""
    pairs = []
    for pair_files in pair_files_list:
        zo_file_path = os.path.join(source_path, pair_files['zo'])
        en_file_path = os.path.join(source_path, pair_files['en'])
        try:
            with open(zo_file_path, 'r', encoding='utf-8') as f_zo, \
                 open(en_file_path, 'r', encoding='utf-8') as f_en:
                zo_lines = [line.strip() for line in f_zo.readlines()]
                en_lines = [line.strip() for line in f_en.readlines()]
                if len(zo_lines) != len(en_lines):
                    print(f"Warning: Mismatch in line count between {pair_files['zo']} and {pair_files['en']}. Skipping.")
                    continue
                loaded = list(zip(zo_lines, en_lines))
                pairs.extend(loaded)
                print(f"  - Loaded {len(loaded)} pairs from {pair_files['zo']}/{pair_files['en']}")
        except FileNotFoundError as e:
            print(f"Error: Could not find data file: {e}. Skipping.")
    return pairs

def load_all_data(config):
    """
    Loads all data pairs from the sources defined in the configuration,
    separating them into training and testing sets.

    Args:
        config (addict.Dict): The loaded project configuration object.

    Returns:
        dict: A dictionary containing 'train' and 'test' lists of sentence pairs.
    """
    all_train_pairs = []
    all_test_pairs = []
    
    if not config.data.sources:
        print("Warning: No data sources found in configuration.")
        return {'train': [], 'test': []}

    print("--- Loading data from sources ---")
    for source_info in config.data.sources:
        source_path = source_info.path
        index_path = os.path.join(source_path, 'index.yaml')

        if not os.path.exists(index_path):
            print(f"Warning: No index.yaml found in source directory: {source_path}. Skipping.")
            continue

        with open(index_path, 'r') as f:
            index_data = yaml.safe_load(f)
        
        print(f"Loading from '{index_data.get('name', source_path)}'...")

        if index_data and index_data.get('type') == 'parallel':
            # FIX: Use .get() to safely access keys.
            train_list = index_data.get('train_pairs', [])
            if train_list:
                print("  Loading training data...")
                all_train_pairs.extend(_load_pairs_from_files(source_path, train_list))
            
            test_list = index_data.get('test_pairs', [])
            if test_list:
                print("  Loading testing data...")
                all_test_pairs.extend(_load_pairs_from_files(source_path, test_list))

    print(f"\nTotal training pairs loaded: {len(all_train_pairs)}")
    print(f"Total testing pairs loaded: {len(all_test_pairs)}")
    return {'train': all_train_pairs, 'test': all_test_pairs}

if __name__ == '__main__':
    # This block demonstrates how the loader works when run directly.
    print("--- Demonstrating zo/sia/data_loader.py ---")
    try:
        from zo.sia.config import load_config
        config = load_config()
        data_sets = load_all_data(config)
        print("\nLoaded data structure:")
        print(data_sets)
    except Exception as e:
        print(f"Error during demonstration: {e}")
