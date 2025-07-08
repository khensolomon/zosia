# zo/sia/data_loader.py
#
# What it does:
# This module is responsible for loading all datasets. It has been updated
# to read the cleaner index.yaml format, where data pairs are specified by
# a simple list of basenames.
#
# How to use it:
# The main script passes the source and target languages to `load_all_data`.
# The function then filters the data sources based on the `use_for_direction`
# key in `config/data.yaml`.

import os
import yaml

def _load_pairs_from_files(source_path, basenames_list):
    """Helper function to load sentence pairs from a list of basenames."""
    pairs = []
    for basename in basenames_list:
        # Construct filenames based on the basename and .zo/.en extensions
        zo_file_path = os.path.join(source_path, f"{basename}.zo")
        en_file_path = os.path.join(source_path, f"{basename}.en")
        
        try:
            with open(zo_file_path, 'r', encoding='utf-8') as f_zo, \
                 open(en_file_path, 'r', encoding='utf-8') as f_en:
                
                zo_lines = [line.strip() for line in f_zo.readlines()]
                en_lines = [line.strip() for line in f_en.readlines()]
                
                if len(zo_lines) != len(en_lines):
                    print(f"Warning: Mismatch in line count between {zo_file_path} and {en_file_path}. Skipping.")
                    continue
                
                # The canonical order is always (zolai, english)
                loaded = list(zip(zo_lines, en_lines))
                pairs.extend(loaded)
                print(f"  - Loaded {len(loaded)} pairs from basename '{basename}'")
        except FileNotFoundError:
            print(f"Error: Could not find data files for basename '{basename}' at {source_path}. Skipping.")
    return pairs

def load_all_data(config, source_lang, target_lang):
    """
    Loads all relevant data pairs from the sources defined in the configuration,
    filtering by the specified training direction.
    """
    all_train_pairs = []
    all_test_pairs = []
    
    training_direction = f"{source_lang}-{target_lang}"
    print(f"--- Loading data for training direction: {training_direction} ---")

    if not config.data.sources:
        print("Warning: No data sources found in configuration.")
        return {'train': [], 'test': []}

    for source_info in config.data.sources:
        source_direction = source_info.get('use_for_direction')
        if source_direction and source_direction != training_direction:
            print(f"Skipping source '{source_info.name}' (intended for '{source_direction}' run).")
            continue

        source_path = source_info.path
        index_path = os.path.join(source_path, 'index.yaml')

        if not os.path.exists(index_path):
            print(f"Warning: No index.yaml found in source directory: {source_path}. Skipping.")
            continue

        with open(index_path, 'r') as f:
            index_data = yaml.safe_load(f)
        
        print(f"Loading from '{index_data.get('name', source_path)}'...")

        if index_data and index_data.get('type') == 'parallel':
            if 'train_pairs' in index_data:
                all_train_pairs.extend(_load_pairs_from_files(source_path, index_data['train_pairs']))
            if 'test_pairs' in index_data:
                all_test_pairs.extend(_load_pairs_from_files(source_path, index_data['test_pairs']))

    print(f"\nTotal training pairs loaded: {len(all_train_pairs)}")
    print(f"Total testing pairs loaded: {len(all_test_pairs)}")
    
    # FIX: Return the actual data that was loaded.
    return {'train': all_train_pairs, 'test': all_test_pairs}
