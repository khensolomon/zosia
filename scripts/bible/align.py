#!/usr//bin/env python3
# -*- coding: utf-8 -*-

"""
Bible Verse Aligner
===================

Description:
------------
This script aligns verses from a 'collection' JSON file with a 'whole bible' JSON file.
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
python align.py \
    --collection my_verses.json \
    --bible kjv.json \
    --source-out english_source.txt \
    --target-out zomi_target.txt

# To output a single TSV file with headers:
python align.py \
    --collection my_verses.json \
    --bible kjv.json \
    --output aligned.tsv \
    --delimiter "tab" \
    --headers "English" "Zomi"
"""

import json
import os
import sys
import argparse
import csv
# Import the shared components from the new config library
from config import BOOK_NAME_TO_NUMBER_MAP, parse_verse_id

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
    # Handle 'tab' as a special case for the delimiter
    if delimiter.lower() == 'tab':
        delimiter = '\t'
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        # Write the user-defined header row
        if headers:
            writer.writerow(headers)
        # Write all the verse pairs
        writer.writerows(verse_pairs)

def align_verses(collection_file, whole_bible_file):
    """
    Processes the input files and returns a list of valid [source, target] verse pairs.
    """
    print("Starting the Bible verse alignment process...")
    verse_pairs = []

    print(f"Loading whole bible file: {whole_bible_file}")
    try:
        with open(whole_bible_file, 'r', encoding='utf-8') as f:
            whole_bible_data = json.load(f).get('book', {})
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{whole_bible_file}'. Please check its format.")
        return None

    print(f"Reading collection file and aligning verses: {collection_file}")
    try:
        with open(collection_file, 'r', encoding='utf-8') as f:
            collection_data = json.load(f)
        
        for verse_data in collection_data.get('raw', []):
            source_text = verse_data.get("text", "")
            verse_id_from_collection = verse_data.get('id')
            target_text = ""
            is_merged = False

            if verse_id_from_collection:
                book_name, chapter, verse = parse_verse_id(verse_id_from_collection)
                
                if book_name and chapter and verse:
                    book_num = BOOK_NAME_TO_NUMBER_MAP.get(book_name)
                    verse_id_log = f"{book_name}.{chapter}:{verse}"
                    target_verses_in_chapter = whole_bible_data.get(book_num, {}).get('chapter', {}).get(chapter, {}).get('verse', {})

                    if target_verses_in_chapter:
                        target_text = target_verses_in_chapter.get(verse, {}).get('text', "")
                        if not target_text:
                            for v_num, v_data in target_verses_in_chapter.items():
                                if str(v_data.get('merge')) == verse:
                                    is_merged = True
                                    break
                            if not is_merged:
                                for i in range(int(verse) - 1, 0, -1):
                                    fallback_text = target_verses_in_chapter.get(str(i), {}).get('text', "")
                                    if fallback_text:
                                        target_text = fallback_text
                                        break
            
            if source_text.strip() and target_text.strip():
                verse_pairs.append([source_text, target_text])

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{collection_file}'. Please check its format.")
        return None

    return verse_pairs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Aligns verses from a collection JSON with a whole bible JSON.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Input Files ---
    parser.add_argument('--collection', required=True, help="Path to the input collection JSON file.")
    parser.add_argument('--bible', required=True, help="Path to the input whole bible JSON file.")

    # --- Output Mode ---
    output_group = parser.add_mutually_exclusive_group(required=True)
    # Mode 1: Two separate files
    output_group.add_argument('--source-out', help="Path for the source text file (requires --target-out).")
    # Mode 2: Single structured file
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
    if args.output and (args.source_out or args.target_out):
         parser.error("Cannot use --output with --source-out or --target-out.")

    # --- File Existence Validation ---
    if not os.path.exists(args.collection):
        sys.exit(f"Error: The specified collection file was not found: {args.collection}")
    if not os.path.exists(args.bible):
        sys.exit(f"Error: The specified bible file was not found: {args.bible}")

    # --- Main Execution ---
    aligned_pairs = align_verses(args.collection, args.bible)

    if aligned_pairs is not None:
        if args.output:
            # Single file output mode
            write_single_file(aligned_pairs, args.output, args.delimiter, args.headers)
        else:
            # Separate files output mode
            source_texts = [pair[0] for pair in aligned_pairs]
            target_texts = [pair[1] for pair in aligned_pairs]
            write_separate_files(source_texts, target_texts, args.source_out, args.target_out)
        
        print("\nProcess finished successfully!")
        print(f"Total verse pairs written: {len(aligned_pairs)}")
