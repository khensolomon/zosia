#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bible Parallel Text Creator
===========================

Description:
------------
This script creates two parallel text files from two different "whole bible" JSON
files (e.g., two different translations or versions).

The script iterates through every verse of the '--source-bible' and writes its text
to the '--source-out' file. For each verse, it looks up the identical verse
(same book, chapter, and verse number) in the '--target-bible' and writes its
text to the '--target-out' file.

If a verse exists in the source but not in the target, the script will search
backwards for the closest preceding verse in the same chapter to use as a
fallback.

Final Validation: After finding a source and target text, the script checks if either
is empty. If so, it prints a warning and skips the pair entirely, excluding it from
the final output files to ensure data quality.

This script imports shared components from '_bible_core.py', which must be in
the same directory.

Usage:
------
Syntax:
python bible_parallel.py --source-bible <path> --target-bible <path> --source-out <path> --target-out <path>

Example Command:
----------------
python bible_parallel.py \\
    --source-bible english_bible.json \\
    --target-bible zomi_bible.json \\
    --source-out english_full.txt \\
    --target-out zomi_full.txt
"""

import json
import os
import sys
import argparse
from _bible_core import BOOK_NUMBER_TO_NAME_MAP # Import the necessary components

def create_parallel_bible_files(source_bible_file, target_bible_file, output_source_file, output_target_file):
    """
    Creates two aligned text files from two different whole bible JSON files.
    """
    print("Starting the Bible parallel text creation process...")

    # 1. Load both bible files into memory for processing.
    print(f"Loading source bible: {source_bible_file}")
    try:
        with open(source_bible_file, 'r', encoding='utf-8') as f:
            source_bible_data = json.load(f).get('book', {})
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{source_bible_file}'. Please check its format.")
        return

    print(f"Loading target bible: {target_bible_file}")
    try:
        with open(target_bible_file, 'r', encoding='utf-8') as f:
            target_bible_data = json.load(f).get('book', {})
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{target_bible_file}'. Please check its format.")
        return

    source_texts = []
    target_texts = []

    # 2. Iterate through the source bible canonically.
    # We sort the book numbers numerically to ensure correct order.
    sorted_book_keys = sorted(source_bible_data.keys(), key=int)

    for book_num in sorted_book_keys:
        book_data = source_bible_data[book_num]
        book_name = BOOK_NUMBER_TO_NAME_MAP.get(book_num, f"Book {book_num}")
        print(f"Processing: {book_name}")
        
        sorted_chapter_keys = sorted(book_data.get('chapter', {}).keys(), key=int)
        for chap_num in sorted_chapter_keys:
            chap_data = book_data['chapter'][chap_num]
            
            sorted_verse_keys = sorted(chap_data.get('verse', {}).keys(), key=int)
            for verse_num in sorted_verse_keys:
                verse_data = chap_data['verse'][verse_num]
                verse_id = f"{book_name}.{chap_num}:{verse_num}"
                
                # First, find both source and target text
                source_text = verse_data.get("text", "")
                target_text = ""

                # Get the target chapter's verses dictionary, if it exists
                target_verses_in_chapter = target_bible_data.get(book_num, {}).get('chapter', {}).get(chap_num, {}).get('verse', {})

                if target_verses_in_chapter:
                    # 1. Try a direct lookup first
                    target_text = target_verses_in_chapter.get(verse_num, {}).get('text', "")

                    # 2. If direct lookup fails, search backwards for the closest preceding verse
                    if not target_text:
                        for i in range(int(verse_num) - 1, 0, -1):
                            fallback_verse_num_str = str(i)
                            fallback_text = target_verses_in_chapter.get(fallback_verse_num_str, {}).get('text', "")
                            if fallback_text:
                                target_text = fallback_text
                                print(f"Warning: Verse '{verse_id}' not found in target. Using closest preceding verse '{book_name}.{chap_num}:{fallback_verse_num_str}' as fallback.")
                                break
                
                # --- New Final Check ---
                # After finding both, check if they are valid before appending.
                if source_text.strip() and target_text.strip():
                    source_texts.append(source_text)
                    target_texts.append(target_text)
                else:
                    if not source_text.strip():
                        print(f"Warning: Source text for ID '{verse_id}' is empty. Skipping verse pair.")
                    else: # This means the target_text must be the empty one
                        print(f"Info: Target for ID '{verse_id}' is empty after all checks. Skipping verse pair.")


    # 3. Write the output files.
    print(f"Writing source text to: {output_source_file}")
    with open(output_source_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(source_texts))

    print(f"Writing target text to: {output_target_file}")
    with open(output_target_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(target_texts))

    print("\nProcess finished successfully!")
    print(f"Total verse pairs written: {len(source_texts)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Creates parallel text files from two different 'whole bible' JSON versions.",
        epilog="Example: python %(prog)s --source-bible eng.json --target-bible zom.json --source-out eng.txt --target-out zom.txt",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--source-bible', required=True, help="Path to the source 'whole bible' JSON file.")
    parser.add_argument('--target-bible', required=True, help="Path to the target 'whole bible' JSON file.")
    parser.add_argument('--source-out', required=True, help="Path for the output source text file.")
    parser.add_argument('--target-out', required=True, help="Path for the output target text file.")
    
    args = parser.parse_args()

    # --- Input File Validation ---
    if not os.path.exists(args.source_bible):
        print(f"Error: The specified source bible file was not found: {args.source_bible}")
        sys.exit(1)

    if not os.path.exists(args.target_bible):
        print(f"Error: The specified target bible file was not found: {args.target_bible}")
        sys.exit(1)

    create_parallel_bible_files(
        args.source_bible,
        args.target_bible,
        args.source_out,
        args.target_out
    )
