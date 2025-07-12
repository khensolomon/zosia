#!/usr/bin/env python3
"""
quotes_merge.py

This script:
- Merges multiple JSON quote files from a folder.
- Avoids duplicate quotes based on text only.
- Merges and normalizes tags and categories.
- Updates tag indices in each quote correctly.
- Assigns new sequential string IDs to each quote.
- Outputs:
    1. A JSON file with merged quotes.
    2. A TXT file with just quote texts.

Usage:
    python quotes_merge.py [input_folder] [output_json] [output_txt]

Defaults:
    input_folder = "./quotes"
    output_json = "merged_quotes.json"
    output_txt = "merged_quotes.txt"
"""

import os
import sys
import json
from glob import glob

DEFAULT_INPUT_FOLDER = './quotes'
DEFAULT_OUTPUT_JSON = 'merged_quotes.json'
DEFAULT_OUTPUT_TXT = 'merged_quotes.txt'

def merge_json_files(folder):
    global_tags = []
    global_categories = []
    seen_texts = set()
    quote_sources = []

    for filepath in glob(os.path.join(folder, '*.json')):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

        local_tags = data.get("tags", [])
        local_categories = data.get("category", [])

        for cat in local_categories:
            if cat not in global_categories:
                global_categories.append(cat)

        quotes = data.get("raw", [])
        for i, quote in enumerate(quotes, 1):
            quote_text_key = quote.get("text", "").strip().lower()
            if not quote_text_key or quote_text_key in seen_texts:
                continue
            seen_texts.add(quote_text_key)

            quote_sources.append((quote, local_tags, filepath, i))

    # Build global tag list
    for _, local_tags, _, _ in quote_sources:
        for tag in local_tags:
            if tag not in global_tags:
                global_tags.append(tag)

    # Remap and assign new IDs
    final_quotes = []
    for new_id, (quote, local_tags, filename, line_no) in enumerate(quote_sources, 1):
        new_tag_indices = []

        for local_index in quote.get("tags", []):
            if isinstance(local_index, int) and 0 <= local_index < len(local_tags):
                tag_name = local_tags[local_index]
                try:
                    global_index = global_tags.index(tag_name)
                    new_tag_indices.append(global_index)
                except ValueError:
                    print(f"Warning: tag '{tag_name}' from file {filename}, line {line_no} not found in global tags.")
            else:
                print(f"Warning: invalid tag index {local_index} in quote: {quote.get('text', '')[:50]}... File: {filename}, line: {line_no}")

        final_quotes.append({
            "id": str(new_id),  # Assigned new sequential string ID
            "text": quote.get("text", ""),
            "author": quote.get("author", ""),
            "date": quote.get("date", ""),
            "note": quote.get("note", ""),
            "tags": new_tag_indices
        })

    return {
        "name": "Merged Quotes Collection",
        "category": global_categories,
        "tags": global_tags,
        "desc": "Merged from multiple files.",
        "raw": final_quotes
    }

def write_outputs(data, json_path, txt_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    with open(txt_path, 'w', encoding='utf-8') as f:
        for quote in data["raw"]:
            f.write(quote["text"].strip() + '\n')

if __name__ == '__main__':
    args = sys.argv[1:]
    input_folder = args[0] if len(args) > 0 else DEFAULT_INPUT_FOLDER
    output_json = args[1] if len(args) > 1 else DEFAULT_OUTPUT_JSON
    output_txt = args[2] if len(args) > 2 else DEFAULT_OUTPUT_TXT

    if not os.path.isdir(input_folder):
        print(f"Input folder does not exist: {input_folder}")
        sys.exit(1)

    merged_data = merge_json_files(input_folder)
    write_outputs(merged_data, output_json, output_txt)

    print("✅ Merge completed.")
    print(f"➡️  Total quotes   : {len(merged_data['raw'])}")
    print(f"➡️  Total tags     : {len(merged_data['tags'])}")
    print(f"➡️  Total categories: {len(merged_data['category'])}")
    print(f"📄 JSON output     : {output_json}")
    print(f"📜 Text output     : {output_txt}")
