# zo/sia/detector.py
#
# What it does:
# This module provides a `LanguageDetector` class that identifies whether a
# given text is English or Zolai. It can be used as a library or run directly
# from the command line to test specific text.
#
# How to use it:
# As a library:
#   from zo.sia.detector import LanguageDetector
#   detector = LanguageDetector()
#   lang = detector.detect("your sentence here")
#
# From the command line:
#   # Run a specific test
#   python -m zo.sia.detector --text "na dam hiam"
#
#   # Run the default demonstration
#   python -m zo.sia.detector

import os
import json
import re
import argparse

class LanguageDetector:
    """
    Detects the language of a text snippet using character n-gram profiles.
    """
    def __init__(self, profile_directory="./data/locale/", lang_codes=['en', 'zo']):
        """
        Initializes the detector by loading language profiles from disk.
        """
        self.profiles = {}
        self.lang_codes = lang_codes
        
        # Suppress the "Initializing..." message if used as a library
        is_main_script = __name__ == '__main__'
        if is_main_script:
            print("Initializing Language Detector...")

        for code in self.lang_codes:
            profile_path = os.path.join(profile_directory, f"{code}.profile.json")
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                    self.profiles[code] = {ngram: i for i, ngram in enumerate(profile_data['ngrams'])}
                if is_main_script:
                    print(f"  - Profile for '{code}' loaded successfully.")
            except FileNotFoundError:
                if is_main_script:
                    print(f"  - Warning: Profile not found for '{code}' at {profile_path}.")
                    print("  - Please run 'scripts/build_profiles.py' to generate it.")
        
        if not self.profiles:
            raise RuntimeError("No language profiles were loaded. Cannot perform detection.")

    def _generate_text_profile(self, text):
        """Generates an n-gram profile for a given text snippet."""
        text = ' ' + re.sub(r'[^a-z\s]', '', text.lower()) + ' '
        ngrams = []
        for n in [2, 3]:
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i+n])
        return ngrams

    def detect(self, text):
        """
        Detects the language of the given text.

        Returns:
            str: The detected language code ('en', 'zo') or 'unknown'.
        """
        if not text or not isinstance(text, str) or text.isspace():
            return 'unknown'

        text_profile = self._generate_text_profile(text)
        if not text_profile:
            return 'unknown'

        scores = {}
        for lang_code, lang_profile in self.profiles.items():
            total_distance = 0
            matches = 0
            for ngram in text_profile:
                distance = lang_profile.get(ngram, len(lang_profile))
                total_distance += distance
                if distance < len(lang_profile):
                    matches += 1
            
            if matches == 0:
                scores[lang_code] = float('inf')
            else:
                scores[lang_code] = total_distance / matches
        
        best_lang = min(scores, key=scores.get)
        
        if scores[best_lang] > 300:
            return 'unknown'
            
        return best_lang

# --- Main Execution Block for Demonstration and CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect the language of a given text.")
    # FIX: Add an optional argument for text input
    parser.add_argument('--text', type=str, help="A specific text string to analyze.")
    args = parser.parse_args()

    try:
        detector = LanguageDetector()

        # If the user provides text, analyze it.
        if args.text:
            print(f"\nAnalyzing text: '{args.text}'")
            detected_lang = detector.detect(args.text)
            print(f"--> Detected language: {detected_lang}")
        # Otherwise, run the standard demonstration.
        else:
            print("\n--- Running standard demonstration ---")
            test_sentences = [
                "how are you",
                "na dam hiam",
                "this is a test sentence in english",
                "kei hong paita",
                "zomi",
                "hello world",
                "你好",
                "12345",
            ]
            for sentence in test_sentences:
                detected_lang = detector.detect(sentence)
                print(f"'{sentence}' -> Detected: {detected_lang}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
