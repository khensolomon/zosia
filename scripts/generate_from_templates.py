# scripts/generate_from_templates.py
#
# What it does:
# This script reads template YAML files from a specified directory (e.g.,
# `./data/templates/`). It generates parallel sentence pairs by replacing
# placeholders like `<NOUN>` with the provided lists of nouns. The default
# input and output paths are now read from the project configuration.
#
# Why it's used:
# To programmatically generate large amounts of structured training data from
# a small set of templates. This is a very efficient way to teach the model
# grammatical patterns and how to handle variable inputs like names or places.
#
# How to use it:
# Run this script from the project's root directory. It will use the paths
# defined in the config file by default.
#
#   python -m scripts.generate_from_templates

import os
import yaml
import argparse
from itertools import product
from zo.sia.config import load_config

def generate_data(args):
    """
    Reads template YAML files, populates them, and writes the resulting
    parallel data to the output directory.
    """
    print(f"--- Generating Data from Templates in '{args.input_dir}' ---")
    
    try:
        template_files = [f for f in os.listdir(args.input_dir) if f.endswith((".yaml", ".yml"))]
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{args.input_dir}'.")
        return

    if not template_files:
        print("No template files found. Nothing to do.")
        return

    for filename in template_files:
        print(f"\nProcessing template: {filename}...")
        filepath = os.path.join(args.input_dir, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                template_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"  - Error parsing YAML file. Skipping. Error: {e}")
                continue

        nouns = template_data.get('NOUN', [])
        en_templates = template_data.get('en', [])
        zo_templates = template_data.get('zo', [])

        if len(en_templates) != len(zo_templates):
            print(f"  - Warning: Mismatch between number of 'en' ({len(en_templates)}) and 'zo' ({len(zo_templates)}) templates. Skipping.")
            continue

        final_en_sentences = []
        final_zo_sentences = []

        # Iterate through the parallel templates
        for en_template, zo_template in zip(en_templates, zo_templates):
            # If the template contains a placeholder, generate all combinations
            if "<NOUN>" in en_template and "<NOUN>" in zo_template:
                if not nouns:
                    print(f"  - Warning: Template '{en_template}' contains <NOUN> but no NOUN list was provided. Skipping.")
                    continue
                for noun in nouns:
                    final_en_sentences.append(en_template.replace("<NOUN>", noun))
                    final_zo_sentences.append(zo_template.replace("<NOUN>", noun))
            # Otherwise, use the template as a single, static sentence pair
            else:
                final_en_sentences.append(en_template)
                final_zo_sentences.append(zo_template)

        # Write the generated sentences to the output files
        base_name = os.path.splitext(filename)[0]
        en_output_path = os.path.join(args.output_dir, f"{base_name}.en")
        zo_output_path = os.path.join(args.output_dir, f"{base_name}.zo")

        os.makedirs(args.output_dir, exist_ok=True)

        with open(en_output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(final_en_sentences))
        
        with open(zo_output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(final_zo_sentences))
            
        print(f"  - Generated {len(final_en_sentences)} sentence pairs.")
        print(f"  - Saved to {en_output_path} and {zo_output_path}")

    print("\n--- Template generation complete! ---")


if __name__ == "__main__":
    # FIX: Load the configuration to get default paths.
    config = load_config()
    default_input_dir = config.paths.get('templates', './data/templates')
    default_output_dir = config.paths.get('parallel_base', './data/parallel_base')

    parser = argparse.ArgumentParser(description="Generate parallel data from YAML templates.")
    # FIX: Use the paths from the config as the default values.
    parser.add_argument('--input_dir', type=str, default=default_input_dir, help="Directory containing the template YAML files.")
    parser.add_argument('--output_dir', type=str, default=default_output_dir, help="Directory to save the generated parallel data files.")
    
    args = parser.parse_args()
    generate_data(args)
