# scripts/analyze_model.py
#
# What it does:
# This script inspects a trained model checkpoint file (.pth) and prints a
# detailed report, now including training performance, data provenance,
# and versioning information.
#
# Why it's used:
# To provide a complete and transparent history of how a model was created,
# what data it was trained on, and how well it performed.
#
# How to use it:
# Run the script from the project's root directory, specifying the model to analyze.
#
#   # Analyze the default zo->en model
#   python -m scripts.analyze_model --source zo --target en

import torch
import argparse
import os
import time
from datetime import datetime, timedelta

# --- Import Our Custom Modules ---
from zo.sia.config import load_config
from zo.sia.model import EncoderRNN, AttnDecoderRNN

def analyze_model(args):
    """Loads a checkpoint and prints a detailed analysis."""
    
    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint not found at '{args.model}'.")
        return

    print(f"--- Analyzing Model Checkpoint: {os.path.basename(args.model)} ---")

    try:
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint file: {e}")
        return

    # --- File Information ---
    file_stat = os.stat(args.model)
    creation_timestamp = checkpoint.get('creation_timestamp')
    print(f"\n[File Information]")
    print(f"  - Size: {file_stat.st_size / (1024*1024):.2f} MB")
    if creation_timestamp:
        print(f"  - Creation Date: {datetime.fromtimestamp(creation_timestamp).strftime('%Y-%m-%d %H:%M:%S')} (from model)")
    
    # --- Training Metadata ---
    metadata = checkpoint.get('training_metadata', {})
    if metadata:
        print(f"\n[Training Summary]")
        duration = metadata.get('training_duration_sec')
        if duration:
            print(f"  - Training Duration: {str(timedelta(seconds=int(duration)))}")
        print(f"  - Final Loss: {metadata.get('final_loss', 'N/A'):.4f}")
        print(f"  - BLEU Score: {metadata.get('bleu_score', 0.0) * 100:.2f}")
        print(f"  - Git Commit: {metadata.get('git_commit_hash', 'N/A')}")

    # --- Data Provenance ---
    if metadata.get('data_sources'):
        print(f"\n[Data Sources Used]")
        for source_name in metadata['data_sources']:
            print(f"  - {source_name}")

    # --- Saved Metadata ---
    config = checkpoint.get('config')
    input_lang = checkpoint.get('input_lang')
    output_lang = checkpoint.get('output_lang')
    max_length = checkpoint.get('max_length')

    print(f"\n[Saved Model Details]")
    if input_lang and output_lang:
        print(f"  - Translation Direction: {input_lang.name} -> {output_lang.name}")
        print(f"  - Source Vocab Size: {input_lang.n_words} words")
        print(f"  - Target Vocab Size: {output_lang.n_words} words")
    if max_length:
        print(f"  - Max Sequence Length: {max_length}")

    # --- Saved Configuration ---
    if config:
        print(f"\n[Training Configuration]")
        print(f"  - Project: {config.project_name}")
        print(f"  - Attention Used: {config.model.get('attention', 'N/A')}")
        print(f"  - Optimizer: {config.training.initial_training.get('optimizer', 'N/A')}")
        print(f"  - Learning Rate: {config.training.initial_training.get('learning_rate', 'N/A')}")
        print(f"  - Batch Size: {config.training.initial_training.get('batch_size', 'N/A')}")
        print(f"  - Encoder Hidden Size: {config.model.encoder.get('hidden_size', 'N/A')}")

    # --- Compatibility Check ---
    print(f"\n[Compatibility Check]")
    try:
        encoder = EncoderRNN(input_lang.n_words, config.model.encoder.hidden_size)
        if config.model.get('attention', False):
            decoder = AttnDecoderRNN(config.model.decoder.hidden_size, output_lang.n_words, max_length=max_length)
        else:
            from zo.sia.model import DecoderRNN
            decoder = DecoderRNN(config.model.decoder.hidden_size, output_lang.n_words)

        current_encoder_keys = set(encoder.state_dict().keys())
        saved_encoder_keys = set(checkpoint['encoder_state_dict'].keys())
        current_decoder_keys = set(decoder.state_dict().keys())
        saved_decoder_keys = set(checkpoint['decoder_state_dict'].keys())

        if current_encoder_keys == saved_encoder_keys:
            print("  - Encoder architecture: COMPATIBLE")
        else:
            print("  - Encoder architecture: INCOMPATIBLE")
        if current_decoder_keys == saved_decoder_keys:
            print("  - Decoder architecture: COMPATIBLE")
        else:
            print("  - Decoder architecture: INCOMPATIBLE")
    except Exception as e:
        print(f"  - Could not perform compatibility check. Error: {e}")

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a trained model checkpoint file.")
    parser.add_argument('--source', type=str, choices=['zo', 'en'], help="Source language of the model.")
    parser.add_argument('--target', type=str, choices=['zo', 'en'], help="Target language of the model.")
    parser.add_argument('--model', type=str, help="Path to the model checkpoint file. If omitted, it will be inferred from the config.")
    
    args = parser.parse_args()

    if not args.model:
        if not (args.source and args.target):
            print("Error: You must provide either the --model path or both --source and --target.")
        else:
            print("Model path not provided, constructing from config...")
            config = load_config()
            filename = config.training.checkpoint.filename_template.format(src=args.source, tgt=args.target)
            args.model = os.path.join(config.paths.experiments, filename)
            analyze_model(args)
    else:
        analyze_model(args)
