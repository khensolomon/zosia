# scripts/build_profiles.py
#
# What it does:
# This script builds language profiles for the character n-gram based language
# detector. It reads the raw training data for each language, calculates the
# frequency of character n-grams (sequences of letters), and saves the top
# N most frequent n-grams as a JSON profile file.
#
# Why it's used:
# This is the "training" step for our language detector. These generated
# profiles act as a "fingerprint" for each language, allowing the detector
# to quickly and accurately identify a language by its character patterns.
#
# How to use it:
# Run this script from the project's root directory. It will automatically
# find the data and create the profile files in the `data/locale/` directory.
#
#   python ./scripts/build_profiles.py

import os
import json
from collections import Counter
import re

# --- Configuration ---
# Where to find the raw data files
DATA_PATH = "./data/parallel_base/"
# Where to save the generated profile files
PROFILE_PATH = "./data/locale/"
# The languages to build profiles for and their corresponding files
LANGUAGES = {
    "en": ["word.en", "test.en"],
    "zo": ["word.zo", "test.zo"]
}
# N-gram settings
N_GRAM_SIZES = [2, 3] # Use bigrams and trigrams
PROFILE_SIZE = 300 # Keep the top 300 most frequent n-grams

def generate_ngrams(text, n):
    """Generates all n-grams for a given text."""
    # Remove padding and split into words
    words = text.strip().split()
    ngrams = []
    for word in words:
        # Add padding to capture start/end of word n-grams
        padded_word = ' ' + word + ' '
        for i in range(len(padded_word) - n + 1):
            ngrams.append(padded_word[i:i+n])
    return ngrams

def build_profile(lang_code):
    """Builds and saves a language profile."""
    print(f"Building profile for language: '{lang_code}'...")
    
    all_text = ""
    files_to_read = LANGUAGES.get(lang_code)
    if not files_to_read:
        print(f"  - No files configured for '{lang_code}'. Skipping.")
        return

    # 1. Read all data for the language
    for filename in files_to_read:
        filepath = os.path.join(DATA_PATH, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                # Normalize text: lowercase and keep only letters and spaces
                content = f.read().lower()
                content = re.sub(r'[^a-z\s]', '', content)
                all_text += content
            print(f"  - Read data from {filepath}")
        else:
            print(f"  - Warning: File not found at {filepath}. Skipping.")
    
    if not all_text:
        print(f"  - No text data found for '{lang_code}'. Cannot build profile.")
        return

    # 2. Generate all n-grams
    all_ngrams = []
    for n in N_GRAM_SIZES:
        all_ngrams.extend(generate_ngrams(all_text, n))
        
    # 3. Count frequencies
    ngram_counts = Counter(all_ngrams)
    
    # 4. Get the top N most frequent n-grams
    most_common_ngrams = [item[0] for item in ngram_counts.most_common(PROFILE_SIZE)]
    
    # 5. Save the profile to a JSON file
    os.makedirs(PROFILE_PATH, exist_ok=True)
    profile_filepath = os.path.join(PROFILE_PATH, f"{lang_code}.profile.json")
    
    profile_data = {
        "name": lang_code,
        "ngrams": most_common_ngrams
    }
    
    with open(profile_filepath, 'w', encoding='utf-8') as f:
        json.dump(profile_data, f, indent=2)
        
    print(f"  - Profile saved successfully to {profile_filepath}")
    print(f"  - Profile contains {len(most_common_ngrams)} n-grams.")

if __name__ == "__main__":
    print("--- Starting Language Profile Builder ---")
    for lang in LANGUAGES:
        build_profile(lang)
    print("\n--- All profiles built successfully! ---")
