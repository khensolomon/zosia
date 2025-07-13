#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bible Verse Aligner
===================

Description:
------------
This script creates two parallel text files from two different Bible JSON sources:
1. A 'collection' JSON file containing a curated list of verses.
2. A 'whole bible' JSON file containing the complete text of the Bible.

The script reads each verse from the 'collection' file and writes its text to a 'source'
output file. It then finds the corresponding verse in the 'whole bible' file and writes
its text to a 'target' output file.

If a verse from the collection cannot be found, the script has fallback logic:
1. It first checks if the verse was merged into a preceding verse (via a "merge" key).
2. If not merged, it searches backwards for the closest preceding verse in the same
   chapter to use instead.
3. If no fallback is found, a blank line is inserted to maintain alignment.

Final Validation: After finding a source and target text, the script checks if either
is empty. If so, it prints a warning and skips the pair entirely, excluding it from
the final output files to ensure data quality.

This script imports shared components from '_bible_core.py', which must be in
the same directory.

Usage:
------
Syntax:
python bible_align.py --collection <path> --bible <path> --source-out <path> --target-out <path>

Example Command:
----------------
python bible_align.py \\
    --collection my_verses.json \\
    --bible kjv.json \\
    --source-out english_source.txt \\
    --target-out zomi_target.txt
"""

import json
import os
import sys
import argparse
# Import the shared components from the new core library
from _bible_core import BOOK_NAME_TO_NUMBER_MAP, parse_verse_id

def create_aligned_verse_files(collection_file, whole_bible_file, output_source_file, output_target_file):
    """
    Creates two aligned text files from a collection JSON and a whole bible JSON.
    """
    print("Starting the Bible verse alignment process...")

    source_texts = []
    target_texts = []

    # 1. Load the whole bible data into memory for quick lookups.
    print(f"Loading whole bible file: {whole_bible_file}")
    try:
        with open(whole_bible_file, 'r', encoding='utf-8') as f:
            whole_bible_data = json.load(f).get('book', {})
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{whole_bible_file}'. Please check its format.")
        return

    # 2. Process the collection/selection file line by line.
    print(f"Reading collection file and aligning verses: {collection_file}")
    try:
        with open(collection_file, 'r', encoding='utf-8') as f:
            collection_data = json.load(f)
        
        for verse_data in collection_data.get('raw', []):
            # First, determine both the source and target text before deciding to add them.
            source_text = verse_data.get("text", "")
            verse_id_from_collection = verse_data.get('id')
            
            target_text = "" # Default to empty string
            is_merged = False

            if verse_id_from_collection:
                book_name, chapter, verse = parse_verse_id(verse_id_from_collection)
                
                if book_name and chapter and verse:
                    book_num = BOOK_NAME_TO_NUMBER_MAP.get(book_name)
                    verse_id_log = f"{book_name}.{chapter}:{verse}"
                    
                    target_verses_in_chapter = whole_bible_data.get(book_num, {}).get('chapter', {}).get(chapter, {}).get('verse', {})

                    if target_verses_in_chapter:
                        # 1. Try a direct lookup first
                        target_text = target_verses_in_chapter.get(verse, {}).get('text', "")

                        # 2. If direct lookup fails, investigate why
                        if not target_text:
                            # 2a. Check if it was merged into another verse
                            for v_num, v_data in target_verses_in_chapter.items():
                                if str(v_data.get('merge')) == verse:
                                    print(f"Warning: Verse '{verse_id_log}' not found. It was merged into verse {book_name}.{chapter}:{v_num}.")
                                    is_merged = True
                                    break # Found merge info, stop searching

                            # 2b. If not merged, search backwards for the closest preceding verse
                            if not is_merged:
                                for i in range(int(verse) - 1, 0, -1):
                                    fallback_verse_num_str = str(i)
                                    fallback_text = target_verses_in_chapter.get(fallback_verse_num_str, {}).get('text', "")
                                    if fallback_text:
                                        target_text = fallback_text
                                        print(f"Warning: Verse '{verse_id_log}' not found in target. Using closest preceding verse '{book_name}.{chapter}:{fallback_verse_num_str}' as fallback.")
                                        break # Found the closest one, stop searching
                else:
                    print(f"Log: Could not process ID '{verse_id_from_collection}'. Appending a blank line to target.")
            else:
                 print(f"Log: Item in collection found with no 'id' field.")

            # --- New Final Check ---
            # After finding both, check if they are valid before appending.
            # .strip() removes whitespace, so strings with only spaces are considered empty.
            if source_text.strip() and target_text.strip():
                source_texts.append(source_text)
                target_texts.append(target_text)
            else:
                verse_id_log = verse_id_from_collection or "Unknown ID"
                if not source_text.strip():
                    print(f"Warning: Source text for ID '{verse_id_log}' is empty. Skipping verse pair.")
                else: # This means the target_text must be the empty one
                    # The reason for the empty target was already logged, so just note the skip.
                    print(f"Info: Target for ID '{verse_id_log}' is empty. Skipping verse pair.")


    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{collection_file}'. Please check its format.")
        return

    # 4. Write the output files.
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
        description="Aligns verses from a collection JSON with a whole bible JSON, creating parallel text files.",
        epilog="Example: python %(prog)s --collection my_verses.json --bible kjv.json --source-out source.txt --target-out target.txt",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--collection', required=True, help="Path to the input collection JSON file.")
    parser.add_argument('--bible', required=True, help="Path to the input whole bible JSON file.")
    parser.add_argument('--source-out', required=True, help="Path for the output source text file.")
    parser.add_argument('--target-out', required=True, help="Path for the output target text file.")
    
    args = parser.parse_args()

    # --- Input File Validation ---
    if not os.path.exists(args.collection):
        print(f"Error: The specified collection file was not found: {args.collection}")
        sys.exit(1)

    if not os.path.exists(args.bible):
        print(f"Error: The specified bible file was not found: {args.bible}")
        sys.exit(1)

    create_aligned_verse_files(
        args.collection,
        args.bible,
        args.source_out,
        args.target_out
    )
