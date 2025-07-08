# zo/sia/detector.py
#
# What it does:
# This module provides a `LanguageDetector` class that identifies the language
# of a given text. It dynamically supports any languages defined in the
# project's main configuration file.
#
# Why it's used:
# To create a fast, accurate, and reusable component for automatic language
# detection, making other scripts more user-friendly.
#
# How to use it:
# 1. First, ensure language profiles have been generated via `scripts/build_profiles.py`.
# 2. As a library:
#    from zo.sia.detector import LanguageDetector
#    detector = LanguageDetector(supported_languages=['en', 'zo'])
#    lang = detector.detect("your sentence here")
#
# 3. From the command line:
#    python -m zo.sia.detector --text "na dam hiam"

import os
import json
import re
import argparse
from zo.sia.config import load_config

class LanguageDetector:
    def __init__(self, profile_directory="./data/locale/", supported_languages=None):
        self.profiles = {}
        if supported_languages is None:
            raise ValueError("The LanguageDetector must be initialized with a list of supported_languages.")
        
        self.lang_codes = supported_languages
        is_main_script = __name__ == '__main__'
        if is_main_script: print("Initializing Language Detector...")
        
        for code in self.lang_codes:
            profile_path = os.path.join(profile_directory, f"{code}.profile.json")
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                    self.profiles[code] = {ngram: i for i, ngram in enumerate(profile_data['ngrams'])}
                if is_main_script: print(f"  - Profile for '{code}' loaded successfully.")
            except FileNotFoundError:
                if is_main_script: print(f"  - Warning: Profile not found for '{code}' at {profile_path}.")
        
        if not self.profiles:
            raise RuntimeError("No language profiles were loaded. Cannot perform detection.")

    def _generate_text_profile(self, text):
        text = ' ' + re.sub(r'[^a-z\s]', '', text.lower()) + ' '
        ngrams = []
        for n in [2, 3]:
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i+n])
        return ngrams

    def detect(self, text):
        if not text or not isinstance(text, str) or text.isspace(): return 'unknown'
        text_profile = self._generate_text_profile(text)
        if not text_profile: return 'unknown'
        scores = {code: sum(self.profiles[code].get(ngram, len(self.profiles[code])) for ngram in text_profile) / len(text_profile) for code in self.profiles if code in self.profiles}
        if not scores: return 'unknown'
        best_lang = min(scores, key=scores.get)
        return best_lang if scores[best_lang] <= 300 else 'unknown'

if __name__ == "__main__":
    config = load_config()
    supported_langs = config.supported_languages

    parser = argparse.ArgumentParser(description="Detect the language of a given text.")
    parser.add_argument('--text', type=str, help="A specific text string to analyze.")
    args = parser.parse_args()

    try:
        detector = LanguageDetector(supported_languages=supported_langs)
        if args.text:
            detected_lang = detector.detect(args.text)
            print(f"--> Detected language: {detected_lang}")
        else:
            print("\n--- Running standard demonstration ---")
            test_sentences = ["how are you", "na dam hiam", "你好"]
            for sentence in test_sentences:
                print(f"'{sentence}' -> Detected: {detector.detect(sentence)}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
