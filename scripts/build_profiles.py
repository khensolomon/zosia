# scripts/build_profiles.py
#
# What it does:
# This script has been made more robust. It now safely handles cases where
# an index.yaml file might be empty or missing expected keys, preventing crashes.
#
# How to use it:
# The command is the same. It will now run without errors even if some data
# source directories have empty or incomplete index files.
#
#   python -m scripts.build_profiles

import os
import json
from collections import Counter
import re
import yaml
from zo.sia.config import load_config

def generate_ngrams(text, n):
    """Generates all n-grams for a given text."""
    words = text.strip().split()
    ngrams = []
    for word in words:
        padded_word = ' ' + word + ' '
        for i in range(len(padded_word) - n + 1):
            ngrams.append(padded_word[i:i+n])
    return ngrams

def build_profile(lang_code, config):
    """Builds and saves a language profile from all available data sources."""
    print(f"Building profile for language: '{lang_code}'...")
    
    all_text = ""
    
    # --- 1. Read Monolingual Data ---
    lang_mono_path = os.path.join(config.paths.monolingual, lang_code)
    if os.path.isdir(lang_mono_path):
        index_path = os.path.join(lang_mono_path, 'index.yaml')
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = yaml.safe_load(f) or {}
            files_to_read = index_data.get('files', [])
            for filename in files_to_read:
                filepath = os.path.join(lang_mono_path, filename)
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f_mono:
                        all_text += f_mono.read() + "\n"
                    print(f"  - Read monolingual data from {filepath}")

    # --- 2. Read Parallel Data ---
    for source_info in config.data.sources:
        source_path = source_info.path
        index_path = os.path.join(source_path, 'index.yaml')
        if not os.path.exists(index_path):
            continue
            
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = yaml.safe_load(f) or {} # FIX: Default to empty dict
            
        if index_data.get('type') == 'parallel':
            for pair_list_key in ['train_pairs', 'test_pairs']:
                # FIX: Use .get() with an empty list as default to prevent errors
                for basename in index_data.get(pair_list_key, []):
                    lang_file = f"{basename}.{lang_code}"
                    filepath = os.path.join(source_path, lang_file)
                    if os.path.exists(filepath):
                        with open(filepath, 'r', encoding='utf-8') as f_parallel:
                            all_text += f_parallel.read() + "\n"
                        print(f"  - Read parallel data from {filepath}")

    if not all_text:
        print(f"  - No text data found for '{lang_code}'. Cannot build profile.")
        return

    # --- 3. Clean and Generate N-grams ---
    cleaned_text = re.sub(r'[^a-z\s]', '', all_text.lower())
    all_ngrams = []
    for n in [2, 3]:
        all_ngrams.extend(generate_ngrams(cleaned_text, n))
        
    ngram_counts = Counter(all_ngrams)
    most_common_ngrams = [item[0] for item in ngram_counts.most_common(300)]
    
    profile_path = os.path.join(config.paths.root, "data", "locale")
    os.makedirs(profile_path, exist_ok=True)
    profile_filepath = os.path.join(profile_path, f"{lang_code}.profile.json")
    
    with open(profile_filepath, 'w', encoding='utf-8') as f:
        json.dump({"name": lang_code, "ngrams": most_common_ngrams}, f, indent=2)
        
    print(f"  - Profile saved successfully to {profile_filepath}")

if __name__ == "__main__":
    print("--- Starting Language Profile Builder ---")
    config = load_config()
    supported_langs = config.supported_languages
    for lang in supported_langs:
        build_profile(lang, config)
    print("\n--- All profiles built successfully! ---")
