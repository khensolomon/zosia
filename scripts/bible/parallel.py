#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bible Parallel Text Creator
===========================

Description:
------------
This script creates parallel text files from two different "whole bible" JSON files.
It offers two output modes:
1. Two separate text files (one for source, one for target).
2. A single structured file (e.g., CSV, TSV) with custom headers.

The script includes fallback logic for missing verses and ensures that only pairs
where both source and target text exist are written to the output.

This script imports shared components from 'config.py', which must be in
the same directory.

Usage:
------
# To output two separate text files:
python parallel.py \\
    --source-bible english.json \\
    --target-bible zomi.json \\
    --source-out english.txt \\
    --target-out zomi.txt

# To output a single CSV file with headers:
python parallel.py \\
    --source-bible english.json \\
    --target-bible zomi.json \\
    --output parallel_verses.csv \\
    --delimiter "," \\
    --headers "KJV" "ZIV"
"""
import json
import os
import sys
import argparse
import csv
# Import the shared components from the new config library
from config import BOOK_NUMBER_TO_NAME_MAP

def write_separate_files(source_texts, target_texts, source_path, target_path):
    """Writes the aligned verses to two separate text files."""
    print(f"Writing source text to: {source_path}")
    with open(source_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(source_texts))

    print(f"Writing target text to: {target_path}")
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(target_texts))

def write_single_file(verse_pairs, output_path, delimiter, headers):
    """Writes the aligned verses to a single structured file (CSV, TSV, etc.)."""
    print(f"Writing structured data to: {output_path}")
    if delimiter.lower() == 'tab':
        delimiter = '\t'
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        if headers:
            writer.writerow(headers)
        writer.writerows(verse_pairs)

def create_parallel_verses(source_bible_file, target_bible_file):
    """
    Processes the input files and returns a list of valid [source, target] verse pairs.
    """
    print("Starting the Bible parallel text creation process...")
    verse_pairs = []

    try:
        print(f"Loading source bible: {source_bible_file}")
        with open(source_bible_file, 'r', encoding='utf-8') as f:
            source_bible_data = json.load(f).get('book', {})
        print(f"Loading target bible: {target_bible_file}")
        with open(target_bible_file, 'r', encoding='utf-8') as f:
            target_bible_data = json.load(f).get('book', {})
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode a JSON file. Please check its format. Details: {e}")
        return None

    sorted_book_keys = sorted(source_bible_data.keys(), key=int)
    for book_num in sorted_book_keys:
        book_name = BOOK_NUMBER_TO_NAME_MAP.get(book_num, f"Book {book_num}")
        print(f"Processing: {book_name}")
        
        sorted_chapter_keys = sorted(source_bible_data.get(book_num, {}).get('chapter', {}).keys(), key=int)
        for chap_num in sorted_chapter_keys:
            chap_data = source_bible_data[book_num]['chapter'][chap_num]
            sorted_verse_keys = sorted(chap_data.get('verse', {}).keys(), key=int)

            for verse_num in sorted_verse_keys:
                source_text = chap_data['verse'][verse_num].get("text", "")
                target_text = ""
                target_verses_in_chapter = target_bible_data.get(book_num, {}).get('chapter', {}).get(chap_num, {}).get('verse', {})

                if target_verses_in_chapter:
                    target_text = target_verses_in_chapter.get(verse_num, {}).get('text', "")
                    if not target_text:
                        for i in range(int(verse_num) - 1, 0, -1):
                            fallback_text = target_verses_in_chapter.get(str(i), {}).get('text', "")
                            if fallback_text:
                                target_text = fallback_text
                                break
                
                if source_text.strip() and target_text.strip():
                    verse_pairs.append([source_text, target_text])

    return verse_pairs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Creates parallel text files from two different 'whole bible' JSON versions.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Input Files ---
    parser.add_argument('--source-bible', required=True, help="Path to the source 'whole bible' JSON file.")
    parser.add_argument('--target-bible', required=True, help="Path to the target 'whole bible' JSON file.")

    # --- Output Mode ---
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument('--source-out', help="Path for the source text file (requires --target-out).")
    output_group.add_argument('--output', help="Path for the single structured output file (e.g., .csv, .tsv).")

    # --- Arguments for Separate File Mode ---
    parser.add_argument('--target-out', help="Path for the target text file (requires --source-out).")
    
    # --- Arguments for Single File Mode ---
    parser.add_argument('--delimiter', default=',', help="Delimiter for single file output. Use 'tab' for tabs. Default is comma.")
    parser.add_argument('--headers', nargs=2, metavar=('SOURCE_HEADER', 'TARGET_HEADER'), help="Two header names for the columns in single file output.")
    
    args = parser.parse_args()

    # --- Argument Validation ---
    if args.source_out and not args.target_out:
        parser.error("--source-out requires --target-out.")
    if args.target_out and not args.source_out:
        parser.error("--target-out requires --source-out.")

    # --- File Existence Validation ---
    if not os.path.exists(args.source_bible):
        sys.exit(f"Error: The specified source bible file was not found: {args.source_bible}")
    if not os.path.exists(args.target_bible):
        sys.exit(f"Error: The specified target bible file was not found: {args.target_bible}")

    # --- Main Execution ---
    aligned_pairs = create_parallel_verses(args.source_bible, args.target_bible)

    if aligned_pairs is not None:
        if args.output:
            write_single_file(aligned_pairs, args.output, args.delimiter, args.headers)
        else:
            source_texts = [pair[0] for pair in aligned_pairs]
            target_texts = [pair[1] for pair in aligned_pairs]
            write_separate_files(source_texts, target_texts, args.source_out, args.target_out)
        
        print("\nProcess finished successfully!")
        print(f"Total verse pairs written: {len(aligned_pairs)}")
