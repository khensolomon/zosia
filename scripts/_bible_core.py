#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bible Core Library
==================

Description:
------------
This is a non-executable library file that contains shared components for various
Bible processing scripts. It provides a centralized place for common data and
functions, such as book name-to-number mappings and verse ID parsing logic.

Do not run this script directly. Instead, import its components into other
executable scripts.

Example Import:
---------------
from _bible_core import BOOK_NAME_TO_NUMBER_MAP, parse_verse_id
"""

import re

# This dictionary maps the standard 3-letter book abbreviations to their 
# corresponding numeric key used in the 'whole_bible.json' format.
# This is crucial for cross-referencing between different Bible versions.
BOOK_NAME_TO_NUMBER_MAP = {
    "GEN": "1", "EXO": "2", "LEV": "3", "NUM": "4", "DEU": "5", "JOS": "6", "JDG": "7", "RUT": "8", "1SA": "9",
    "2SA": "10", "1KI": "11", "2KI": "12", "1CH": "13", "2CH": "14", "EZR": "15", "NEH": "16", "EST": "17",
    "JOB": "18", "PSA": "19", "PRO": "20", "ECC": "21", "SNG": "22", "ISA": "23", "JER": "24", "LAM": "25",
    "EZK": "26", "DAN": "27", "HOS": "28", "JOL": "29", "AMO": "30", "OBA": "31", "JON": "32", "MIC": "33",
    "NAM": "34", "HAB": "35", "ZEP": "36", "HAG": "37", "ZEC": "38", "MAL": "39", "MAT": "40", "MRK": "41",
    "LUK": "42", "JHN": "43", "ACT": "44", "ROM": "45", "1CO": "46", "2CO": "47", "GAL": "48", "EPH": "49",
    "PHP": "50", "COL": "51", "1TH": "52", "2TH": "53", "1TI": "54", "2TI": "55", "TIT": "56", "PHM": "57",
    "HEB": "58", "JAS": "59", "1PE": "60", "2PE": "61", "1JN": "62", "2JN": "63", "3JN": "64", "JUD": "65",
    "REV": "66",
    "EZE":"26", "JOEL":"29", "NAH":"34", "PAS":"19"
}

# A reverse map to convert book numbers back to names for creating the standard ID format.
BOOK_NUMBER_TO_NAME_MAP = {v: k for k, v in BOOK_NAME_TO_NUMBER_MAP.items()}


def parse_verse_id(verse_id):
    """
    Parses a verse ID string (e.g., 'Acts.7:15') into its components.
    
    Args:
        verse_id (str): The verse ID string.
        
    Returns:
        tuple: A tuple containing the book name, chapter, and verse, or (None, None, None) if invalid.
    """
    # Use a regular expression to robustly parse the ID format like 'BOOK.CHAPTER:VERSE'
    match = re.match(r'([a-zA-Z0-9]+)\.(\d+):(\d+)', verse_id)
    if match:
        book_short_name, chapter, verse = match.groups()
        # Normalize the book name from the ID to the 3-letter standard (e.g., Acts -> ACT)
        book_short_name_upper = book_short_name.upper()
        for key in BOOK_NAME_TO_NUMBER_MAP:
            if book_short_name_upper.startswith(key):
                return key, chapter, verse
    print(f"Warning: Could not parse verse ID: '{verse_id}'")
    return None, None, None

