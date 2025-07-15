# scripts/back_translate.py
#
# What it does:
# This script has been made more robust. It now safely handles cases where
# an index.yaml file might be empty or missing expected keys, preventing crashes.
#
# How to use it:
# The command is the same. It will now run without errors even if the input
# directory has an empty or incomplete index file.
#
#   python -m scripts.back_translate --source zo --target en

import torch
import argparse
import os
import yaml
from tqdm import tqdm

# --- Import Our Custom Modules ---
from zo.sia.config import load_config
from zo.sia.model import EncoderRNN, DecoderRNN, AttnDecoderRNN
from zo.sia.data_utils import Lang
from zo.sia.translate import Translator

def back_translate(args):
    """Main function to perform the back-translation process."""
    print(f"--- Starting Back-Translation ---")
    print(f"Model: {args.model}")
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")

    try:
        device = torch.device(args.device)
        translator = Translator(args.model, device)
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at '{args.model}'.")
        print(f"Please ensure you have trained this model first: python -m zo.sia.main --source {args.source} --target {args.target}")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    index_path = os.path.join(args.input_dir, 'index.yaml')
    if not os.path.exists(index_path):
        print(f"Error: 'index.yaml' not found in the input directory: {args.input_dir}")
        return
        
    with open(index_path, 'r', encoding='utf-8') as f:
        try:
            index_data = yaml.safe_load(f) or {} # FIX: Default to empty dict
            # FIX: Use .get() with an empty list as default to prevent errors
            input_files = index_data.get('files', [])
            if not input_files:
                 print(f"Warning: No files listed in {index_path}. Nothing to do.")
                 return
        except yaml.YAMLError as e:
            print(f"Error parsing {index_path}: {e}")
            return

    generated_basenames = []

    for filename in input_files:
        print(f"\nProcessing file: {filename}...")
        input_filepath = os.path.join(args.input_dir, filename)

        if not os.path.exists(input_filepath):
            print(f"  - Warning: File '{filename}' listed in index.yaml but not found. Skipping.")
            continue
        
        with open(input_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        original_lines = []
        translated_lines = []

        for line in tqdm(lines, desc=f"Translating {filename}"):
            line = line.strip()
            if not line:
                continue
            
            translation, _ = translator.translate(line)
            
            original_lines.append(line)
            translated_lines.append(translation)

        base_name = os.path.splitext(filename)[0]
        
        output_translated_path = os.path.join(args.output_dir, f"{base_name}.{args.target}")
        output_original_path = os.path.join(args.output_dir, f"{base_name}.{args.source}")

        with open(output_translated_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(translated_lines))
        
        with open(output_original_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(original_lines))
            
        print(f"  - Saved synthetic data to: {output_translated_path}")
        print(f"  - Saved original data to: {output_original_path}")

        generated_basenames.append(base_name)

    index_data = {
        "name": f"synthetic-{args.source}-to-{args.target}",
        "description": f"Synthetic parallel data generated via back-translation from '{args.input_dir}'.",
        "type": "parallel",
        "train_pairs": generated_basenames
    }
    
    index_filepath = os.path.join(args.output_dir, "index.yaml")
    with open(index_filepath, 'w', encoding='utf-8') as f:
        yaml.dump(index_data, f, default_flow_style=False, sort_keys=False)
        
    print(f"\nGenerated index.yaml at: {index_filepath}")
    print("\n--- Back-Translation Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic parallel data using back-translation.")
    parser.add_argument('--source', type=str, required=True, help="The language of the text in the input directory (e.g., 'zo').")
    parser.add_argument('--target', type=str, required=True, help="The language to translate the text into (e.g., 'en').")
    parser.add_argument('--model', type=str, help="Path to the pre-trained model checkpoint. If omitted, it will be inferred from the config.")
    parser.add_argument('--input_dir', type=str, help="Directory containing monolingual files. If omitted, it will be inferred from the config.")
    parser.add_argument('--output_dir', type=str, help="Directory to save the new synthetic data. If omitted, it will be created automatically.")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use for translation ('cpu' or 'cuda').")
    
    args = parser.parse_args()
    config = load_config()

    if not args.model:
        print("Model path not provided, constructing from config...")
        filename = config.training.checkpoint.filename_template.format(src=args.source, tgt=args.target)
        args.model = os.path.join(config.paths.experiments, filename)

    if not args.input_dir:
        print("Input directory not provided, constructing from config...")
        args.input_dir = os.path.join(config.paths.monolingual, args.source)

    if not args.output_dir:
        print("Output directory not provided, constructing automatically...")
        output_dir_name = f"{args.source}-{args.target}"
        args.output_dir = os.path.join(config.paths.data, "synthetic", output_dir_name)

    back_translate(args)
