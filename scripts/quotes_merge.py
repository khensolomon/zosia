"""
Quotes Merge Script

This script merges multiple JSON files containing quotes into a single master JSON file.
It ensures no duplicate entries (based on "text") and remaps tag indices to maintain consistency.
It also exports the quote texts into a plain .txt file for convenience.

Features:
- Supports merging from an existing master collection if found.
- Avoids duplicate entries by checking quote text.
- Remaps tag indices properly based on global tag list.
- Provides a summary of entries before and after merging.
- Keeps "name" and "desc" from the existing collection if it exists.
- Requires no external dependencies (only built-in modules).

Usage:
    python quotes_merge.py [INPUT_FOLDER] [OUTPUT_JSON] [OUTPUT_TXT]
    python ./scripts/quotes_merge.py

Defaults:
    INPUT_FOLDER: "./tmp/quote"
    OUTPUT_JSON: "./data/corpus/quotes.json"
    OUTPUT_TXT: "./tmp/merged_quotes.txt""
"""

import os
import sys
import json

DEFAULT_INPUT_FOLDER = "./tmp/quote"
DEFAULT_OUTPUT_JSON = "./data/corpus/quotes.json"
DEFAULT_OUTPUT_TXT = "./tmp/merged_quotes.txt"


def merge_json_files(input_folder):
    all_tags = []
    all_categories = set()
    quote_text_map = {}

    if os.path.exists(DEFAULT_OUTPUT_JSON):
        with open(DEFAULT_OUTPUT_JSON, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            all_tags = existing_data.get("tags", [])
            all_categories.update(existing_data.get("category", []))
            for q in existing_data.get("raw", []):
                quote_text_map[q["text"]] = q
        print("Loaded existing quotes collection from:", DEFAULT_OUTPUT_JSON)
    else:
        existing_data = None
        print("No existing quotes collection found. Starting from scratch.")

    new_raw = []

    for filename in os.listdir(input_folder):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(input_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {filename}")
                continue

        local_tags = data.get("tags", [])
        local_tag_indices = {tag: all_tags.index(tag) if tag in all_tags else None for tag in local_tags}
        for tag in local_tags:
            if tag not in all_tags:
                all_tags.append(tag)
                local_tag_indices[tag] = len(all_tags) - 1

        all_categories.update(data.get("category", []))

        for q in data.get("raw", []):
            if q["text"] in quote_text_map:
                continue

            try:
                q["tags"] = [local_tag_indices.get(local_tags[i], -1) for i in q.get("tags", []) if 0 <= i < len(local_tags)]
                q["tags"] = [i for i in q["tags"] if i >= 0]
            except Exception as e:
                print(f"Error in {filename} at quote: {q.get('text')} -> {e}")
                continue

            q["id"] = str(len(quote_text_map) + len(new_raw) + 1)
            new_raw.append(q)

    merged_raw = list(quote_text_map.values()) + new_raw

    print("\n--- Merge Summary (Quotes) ---")
    print(f"Existing entries: {len(quote_text_map)}")
    print(f"New entries added: {len(new_raw)}")
    print(f"Total after merge: {len(merged_raw)}\n")

    merged_data = {
        "name": existing_data.get("name") if existing_data else "",
        "category": sorted(list(all_categories)),
        "tags": all_tags,
        "desc": existing_data.get("desc") if existing_data else "??",
        "raw": merged_raw
    }

    return merged_data


def write_outputs(merged_data, output_json, output_txt):
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    with open(output_txt, "w", encoding="utf-8") as f:
        for quote in merged_data["raw"]:
            f.write(quote["text"].strip() + "\n")


if __name__ == "__main__":
    input_folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FOLDER
    output_json = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_JSON
    output_txt = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUTPUT_TXT

    merged_data = merge_json_files(input_folder)
    write_outputs(merged_data, output_json, output_txt)
    print("Quotes merge completed.")
