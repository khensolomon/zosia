#!/usr/bin/env python3
"""
Template-Based Sentence Generator
=================================

This script compiles YAML-defined sentence templates into parallel TSV datasets.

Each template file may include:
- External references to shared `slots.yaml` and `rules.yaml`
- One or more `templates` with slot requirements, output headers, logic rules, and sentence formatting

Usage:
    python generate_sentences.py templates/*.yaml
    python generate_sentences.py templates/basic.yaml --outdir my_outputs/

Arguments:
    template_files     One or more YAML template files to compile

Options:
    --outdir DIR       Directory to save generated TSVs (default: ./generated)

Expected Template YAML Structure:
---------------------------------

include:
  slots: "slots/common_slots.yaml"
  rules: "rules/common_rules.yaml"

slots:      # (Optional override or additions)
rules:      # (Optional override or additions)

templates:
  - name: "template_id"
    slots: ["subject", "verb", "object"]
    output_headers: ["EN", "ZO", "GLOSS"]
    sentence:
      EN: "{subject} {verb} {object}."
      ZO: "{subject} {object} {verb}."
      GLOSS: "{subject} {object} {verb}."
    logic:
      verb:
        lookup: "verb_table"
        key_path: ["subject", "features", "person"]

This tool is modular and declarative. No hardcoded logic.

Author: You + ChatGPT
"""

import yaml
import itertools
import csv
import sys
import argparse
from pathlib import Path


def load_yaml(path):
    """Load a YAML file from the given path."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_by_path(data, path):
    """Traverse nested dictionaries using a list of keys."""
    for key in path:
        data = data[key]
    return data


def merge_dicts(base, override):
    """Merge two dictionaries (non-recursive)."""
    result = dict(base)
    result.update(override or {})
    return result


def load_template_file(template_file):
    """
    Load a self-contained or modular template file.

    Supports `include:` section to pull in external slots/rules YAMLs.
    """
    raw = load_yaml(template_file)
    includes = raw.get("include", {})

    slots = load_yaml(includes["slots"]) if "slots" in includes else {}
    rules = load_yaml(includes["rules"]) if "rules" in includes else {}

    slots = merge_dicts(slots, raw.get("slots", {}))
    rules = merge_dicts(rules, raw.get("rules", {}))
    templates = raw["templates"]

    return slots, rules, templates


def process_template(template, slots, rules, outdir):
    """
    Generate all sentence combinations for a single template,
    then save as a TSV file in the specified output directory.
    """
    used_slots = template["slots"]
    headers = template["output_headers"]
    fmt = template["sentence"]
    logic = template.get("logic", {})
    slot_values = [slots[k] for k in used_slots]

    rows = []

    for combo in itertools.product(*slot_values):
        combo_dict = {used_slots[i]: combo[i] for i in range(len(used_slots))}
        values = {lang: {} for lang in headers}

        for slot, item in combo_dict.items():
            for lang in headers:
                if slot not in logic or lang != "ZO":
                    values[lang][slot] = item["forms"][lang.lower()]

        for slot, rule in logic.items():
            lookup = rules[rule["lookup"]]
            key = get_by_path(combo_dict, rule["key_path"])
            word_id = combo_dict[slot]["id"]
            values["ZO"][slot] = lookup[word_id][key]

        row = [fmt[lang].format(**values[lang]) for lang in headers]
        rows.append(row)

    out_name = Path(outdir) / f"generated_{template['name']}.tsv"
    out_name.parent.mkdir(parents=True, exist_ok=True)

    with open(out_name, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"✅ {out_name} created with {len(rows)} rows.")


def main():
    """Parse CLI args, compile all provided template YAML files."""
    parser = argparse.ArgumentParser(
        description="Compile YAML-based translation templates into TSV datasets."
    )
    parser.add_argument(
        "template_files",
        nargs="+",
        help="YAML template file(s) to process"
    )
    parser.add_argument(
        "--outdir",
        default="generated",
        help="Output directory for .tsv files (default: ./generated)"
    )

    args = parser.parse_args()

    for file_path in args.template_files:
        path = Path(file_path)
        print(f"\n🔧 Processing: {path.name}")
        slots, rules, templates = load_template_file(path)
        for template in templates:
            process_template(template, slots, rules, args.outdir)


if __name__ == "__main__":
    main()
