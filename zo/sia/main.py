# main.py
#
# Project: Sia
# Path: ./zo/sia/main.py
#
# What it does:
# This script is the high-level orchestrator for the model training pipeline.
# It prepares the configuration, data, and models, then hands them off to the
# `Trainer` class to perform the actual training and evaluation. It dynamically
# reads the list of supported languages from the configuration.
#
# Why it's used:
# It provides a single, unified command to train models for any specified
# language direction. Its separation of concerns keeps the top-level logic
# clean and easy to understand.
#
# How to use it:
# Specify the source and target languages from the list in `config/default.yaml`.
#
#   # Train a Zolai to English model
#   python -m zo.sia.main --source zo --target en

import torch
import torch.optim as optim
import argparse
import os
import subprocess

# --- Import Our Custom Modules ---
from zo.sia.config import load_config
from zo.sia.data_loader import load_all_data
from zo.sia.data_utils import normalize_string, prepare_data, calculate_max_length
from zo.sia.model import EncoderRNN, DecoderRNN, AttnDecoderRNN
from zo.sia.trainer import Trainer

def get_git_hash():
    """Gets the current git commit hash, if available."""
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "N/A"

def main():
    config = load_config()
    supported_langs = config.supported_languages

    parser = argparse.ArgumentParser(description="Train a ZoSia translation model.")
    parser.add_argument('--source', type=str, required=True, choices=supported_langs, help="Source language.")
    parser.add_argument('--target', type=str, required=True, choices=supported_langs, help="Target language.")
    args = parser.parse_args()

    if args.source == args.target:
        raise ValueError("Source and target languages cannot be the same.")

    print(f"--- Preparing to train model for {args.source} -> {args.target} ---")
    
    data_sets = load_all_data(config, args.source, args.target)
    train_pairs, test_pairs = data_sets['train'], data_sets['test']
    
    # The data loader returns pairs in a canonical order (e.g., zo-en).
    # This logic swaps them if the training run is the reverse.
    # This assumes the second language in the config is the "base" for pairs.
    if args.source == supported_langs[0]:
        train_pairs = [(p[1], p[0]) for p in train_pairs]
        test_pairs = [(p[1], p[0]) for p in test_pairs]

    if not train_pairs:
        print("No training data loaded. Exiting.")
        return

    input_lang, output_lang, normalized_train_pairs = prepare_data(train_pairs, config.tokenizer.special_tokens)
    normalized_test_pairs = [[normalize_string(p[0]), normalize_string(p[1])] for p in test_pairs]
    max_length = calculate_max_length(normalized_train_pairs + normalized_test_pairs)
    if config.tokenizer.max_length != "auto":
        max_length = int(config.tokenizer.max_length)
    print(f"Input Vocab Size: {input_lang.n_words}, Output Vocab Size: {output_lang.n_words}, Max Length: {max_length}")

    device = torch.device("cuda" if config.model.device == "auto" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = EncoderRNN(input_lang.n_words, config.model.encoder.hidden_size).to(device)
    decoder = AttnDecoderRNN(config.model.decoder.hidden_size, output_lang.n_words, max_length=max_length).to(device) if config.model.get('attention', False) else DecoderRNN(config.model.decoder.hidden_size, output_lang.n_words).to(device)
    
    optimizer_name = config.training.initial_training.optimizer.lower()
    lr = config.training.initial_training.learning_rate
    print(f"Using optimizer: {optimizer_name} with learning rate: {lr}")

    if optimizer_name == 'adam':
        encoder_optimizer, decoder_optimizer = optim.Adam(encoder.parameters(), lr=lr), optim.Adam(decoder.parameters(), lr=lr)
    else:
        encoder_optimizer, decoder_optimizer = optim.SGD(encoder.parameters(), lr=lr), optim.SGD(decoder.parameters(), lr=lr)

    trainer = Trainer(config, encoder, decoder, encoder_optimizer, decoder_optimizer, device)
    duration, final_loss = trainer.train(normalized_train_pairs, input_lang, output_lang, max_length)
    
    bleu_score = trainer.evaluate(normalized_test_pairs, input_lang, output_lang, max_length) if normalized_test_pairs else 0.0

    training_metadata = {
        'training_duration_sec': duration, 'final_loss': final_loss, 'bleu_score': bleu_score,
        'data_sources': [source['name'] for source in config.data.sources if not source.get('use_for_direction') or source.get('use_for_direction') == f"{args.source}-{args.target}"],
        'git_commit_hash': get_git_hash()
    }

    filename = config.training.checkpoint.filename_template.format(src=args.source, tgt=args.target)
    trainer.save_checkpoint(filename, input_lang, output_lang, max_length, training_metadata)

if __name__ == '__main__':
    main()
