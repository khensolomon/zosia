# main.py
#
# Project: Sia
# Path: ./zo/sia/main.py
#
# What it does:
# This script orchestrates the model training pipeline. It has been updated
# to use a configurable optimizer (Adam or SGD) for better training stability.

import torch
import torch.optim as optim
import argparse
import os

# --- Import Our Custom Modules ---
from zo.sia.config import load_config
from zo.sia.data_loader import load_all_data
from zo.sia.data_utils import normalize_string, prepare_data, calculate_max_length
from zo.sia.model import EncoderRNN, DecoderRNN, AttnDecoderRNN
from zo.sia.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train a ZoSia translation model.")
    parser.add_argument('--source', type=str, required=True, choices=['zo', 'en'], help="Source language.")
    parser.add_argument('--target', type=str, required=True, choices=['zo', 'en'], help="Target language.")
    args = parser.parse_args()

    if args.source == args.target:
        raise ValueError("Source and target languages cannot be the same.")

    print(f"--- Preparing to train model for {args.source} -> {args.target} ---")

    config = load_config()
    data_sets = load_all_data(config)
    
    train_pairs = data_sets['train']
    test_pairs = data_sets['test']
    
    if args.source == 'en':
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

    device_str = config.model.device
    device = torch.device("cuda" if device_str == "auto" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = EncoderRNN(input_lang.n_words, config.model.encoder.hidden_size).to(device)
    
    if config.model.get('attention', False):
        decoder = AttnDecoderRNN(config.model.decoder.hidden_size, output_lang.n_words, max_length=max_length).to(device)
    else:
        decoder = DecoderRNN(config.model.decoder.hidden_size, output_lang.n_words).to(device)
    
    # FIX: Use the optimizer specified in the config file.
    optimizer_name = config.training.initial_training.optimizer.lower()
    lr = config.training.initial_training.learning_rate
    print(f"Using optimizer: {optimizer_name} with learning rate: {lr}")

    if optimizer_name == 'adam':
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=lr)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")

    trainer = Trainer(config, encoder, decoder, encoder_optimizer, decoder_optimizer, device)
    trainer.train(normalized_train_pairs, input_lang, output_lang, max_length)
    
    if normalized_test_pairs:
        trainer.evaluate(normalized_test_pairs, input_lang, output_lang, max_length)

    filename = config.training.checkpoint.filename_template.format(src=args.source, tgt=args.target)
    trainer.save_checkpoint(filename, input_lang, output_lang, max_length)

if __name__ == '__main__':
    main()
