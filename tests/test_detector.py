# tests/test_detector.py
#
# What it does:
# This file contains unit tests for the LanguageDetector class. It has been
# updated to correctly initialize the detector with a list of supported
# languages, as is now required.
#
# How to run it:
# Run pytest from the project's root directory:
#   pytest tests/test_detector.py

import pytest
import json
import os
from zo.sia.detector import LanguageDetector

@pytest.fixture
def temp_profile_dir(tmp_path):
    """A pytest fixture to create a temporary directory with dummy language profiles."""
    locale_dir = tmp_path / "locale"
    locale_dir.mkdir()

    # Create a simple English profile
    en_profile = {
        "name": "en",
        "ngrams": [" th", "he ", "ing", "er", " an", "re", " on", "at", "en"]
    }
    with open(locale_dir / "en.profile.json", "w") as f:
        json.dump(en_profile, f)

    # Create a simple Zolai profile
    zo_profile = {
        "name": "zo",
        "ngrams": [" ah", "na ", "ia", "ei", " om", "an", "in", "kh", "ta"]
    }
    with open(locale_dir / "zo.profile.json", "w") as f:
        json.dump(zo_profile, f)
        
    return str(locale_dir)

# A list of test cases for parametrization
detection_test_cases = [
    ("this is a test in english", "en"),
    ("hello there, how are you?", "en"),
    ("na dam hiam", "zo"),
    ("kei hong paita", "zo"),
    ("a inn ah a om", "zo"),
    ("你好世界", "unknown"),  # Non-latin text
    ("12345 !@#$", "unknown"), # No valid n-grams
    ("", "unknown"),          # Empty string
]

@pytest.mark.parametrize("text, expected_lang", detection_test_cases)
def test_language_detection(temp_profile_dir, text, expected_lang):
    """
    Tests the LanguageDetector with various inputs.
    """
    # FIX: Pass the list of supported languages when initializing the detector.
    detector = LanguageDetector(
        profile_directory=temp_profile_dir, 
        supported_languages=['en', 'zo']
    )
    
    detected_lang = detector.detect(text)
    
    assert detected_lang == expected_lang

def test_detector_no_profiles(tmp_path):
    """
    Tests that the detector raises an error if no profiles are found.
    """
    with pytest.raises(RuntimeError):
        # We still expect a RuntimeError if the profile files themselves are missing.
        LanguageDetector(
            profile_directory=str(tmp_path),
            supported_languages=['en', 'zo']
        )

def test_detector_no_supported_languages():
    """
    Tests that the detector raises a ValueError if no languages are provided.
    """
    with pytest.raises(ValueError):
        LanguageDetector(supported_languages=None)
