# v.0016
"""
Zolai-Centric Multilingual NMT CLI Tool

This script provides a command-line interface for training a sequence-to-sequence
NMT model with an Attention mechanism. It is designed to be a flexible framework
for translating between Zolai (zo) and multiple other languages.

Features:
- Data Augmentation: Generates training data from YAML templates for richer datasets.
- Template Testing: A dedicated CLI command to test the template engine's output.
- Zolai-centric design: 'zo' must be either the source or target language.
- Flexible multi-language support defined in a simple configuration.
- Efficient checkpointing: Saves only the latest model per language pair.
- Encoder-Decoder architecture with Luong Attention.
- Attention-based copy mechanism for handling unknown (<unk>) tokens.
- Vocabulary and model parameters saved together in checkpoints.
- Compatible with PyTorch 2.6+ secure loading.

Usage examples:
  # Test the template engine and save the output to ./tmp/
  python ./scripts/nmt.py test-template --source zo --target en

  # Train a model using both TSV files and YAML templates
  python ./scripts/nmt.py train --source zo --target en --use-templates

  # Translate a text
  python ./scripts/nmt.py translate --source zo --target en --text "Solomon hong paita"
"""
import os
import io
import random
import argparse
import sys
import re
import itertools
import yaml # Requires PyYAML
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.serialization import safe_globals
from collections import Counter

# --- 1. Configuration and Constants ---

# Special tokens
SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN = 0, 1, 2, 3

# Language and Path Configuration
PRIMARY_LANGUAGE = "zo"
SUPPORTED_LANGUAGES = ["zo", "en", "no"]
CORPUS_DIR = "./data/corpus"
TEMPLATE_DIR = "./data/templates"
CHECKPOINT_DIR = "./experiments"
TMP_DIR = "./tmp" # Directory for temporary files
CHECKPOINT_FILE_PATTERN = "checkpoint_{src}-{tar}.pt"

# Model and Training Parameters
CONFIG = {
    'hidden_size': 256, 'learning_rate': 0.01, 'epochs': 75000,
    'print_every': 5000, 'vocab_size': 4000, 'min_count': 1,
}


# --- 2. Template-based Data Augmentation ---

def generate_from_template_file(file_path, source_lang, target_lang):
    """
    Generates sentence pairs from a single YAML template file.
    Skips the file if it doesn't contain templates for both the source and target languages.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            template_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Warning: Could not parse YAML file {file_path}. Error: {e}")
        return []

    # Check if the required language keys exist in the template file. If not, skip.
    if source_lang not in template_data or target_lang not in template_data:
        return []

    source_templates = template_data.get(source_lang, [])
    target_templates = template_data.get(target_lang, [])

    if len(source_templates) != len(target_templates):
        print(f"Warning: Mismatch in number of templates for '{source_lang}' and '{target_lang}' in {file_path}. Skipping file.")
        return []

    generated_pairs = []
    for src_template, tar_template in zip(source_templates, target_templates):
        # Find all unique placeholders from both source and target templates
        placeholders = set(re.findall(r"<(\w+)>", src_template) + re.findall(r"<(\w+)>", tar_template))
        
        if not all(p in template_data for p in placeholders):
            print(f"Warning: Template pair '{src_template}' / '{tar_template}' has undefined placeholders. Skipping.")
            continue
        
        if not placeholders:
            generated_pairs.append([src_template, tar_template])
            continue
        
        value_lists = [template_data[p] for p in placeholders]
        
        for combo in itertools.product(*value_lists):
            src_sentence, tar_sentence = src_template, tar_template
            for i, placeholder in enumerate(placeholders):
                # Note: This simple replace might not work if placeholder names are substrings of others.
                # For this project's scope, it's acceptable.
                src_sentence = src_sentence.replace(f"<{placeholder}>", combo[i])
                tar_sentence = tar_sentence.replace(f"<{placeholder}>", combo[i])
            generated_pairs.append([src_sentence, tar_sentence])
            
    return generated_pairs

def load_templated_data(source_lang, target_lang):
    """Loads all data from YAML templates in the template directory."""
    if not os.path.exists(TEMPLATE_DIR):
        return []
        
    all_pairs = []
    for filename in os.listdir(TEMPLATE_DIR):
        if filename.endswith((".yaml", ".yml")):
            file_path = os.path.join(TEMPLATE_DIR, filename)
            all_pairs.extend(generate_from_template_file(file_path, source_lang, target_lang))
    return all_pairs


# --- 3. Data Preparation and Vocabulary ---

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<SOS>": SOS_TOKEN, "<EOS>": EOS_TOKEN, "<PAD>": PAD_TOKEN, "<UNK>": UNK_TOKEN}
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.word_count = Counter()
        self.n_words = 4

    def add_sentence(self, sentence):
        self.word_count.update(sentence.split(' '))

    def trim_vocab(self, max_size, min_count):
        filtered_words = [w for w, c in self.word_count.items() if c >= min_count]
        filtered_words.sort(key=lambda w: self.word_count[w], reverse=True)
        kept_words = filtered_words[:max_size - self.n_words]
        self.word2index = {"<SOS>": SOS_TOKEN, "<EOS>": EOS_TOKEN, "<PAD>": PAD_TOKEN, "<UNK>": UNK_TOKEN}
        self.index2word = {v: k for k, v in self.word2index.items()}
        for word in kept_words:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

def read_data(file_path):
    with io.open(file_path, encoding='utf-8') as f:
        return [line.strip().split('\t') for line in f if '\t' in line and "sentence" not in line]

def prepare_data(source_lang, target_lang, vocab_size, min_count, use_templates=False):
    file_path = os.path.join(CORPUS_DIR, f"{source_lang}-{target_lang}.tsv")
    pairs = read_data(file_path)
    
    if use_templates:
        print("Generating additional data from templates...")
        template_pairs = load_templated_data(source_lang, target_lang)
        print(f"Generated {len(template_pairs)} pairs from templates.")
        pairs.extend(template_pairs)

    if not pairs:
        raise ValueError(f"No data found for the pair '{source_lang}-{target_lang}'.")

    input_lang, output_lang = Lang(source_lang), Lang(target_lang)
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    input_lang.trim_vocab(vocab_size, min_count)
    output_lang.trim_vocab(vocab_size, min_count)

    max_len = max(len(s.split(' ')) for p in pairs for s in p) + 2
    return input_lang, output_lang, pairs, max_len

def indexes_from_sentence(lang, sentence):
    return [lang.word2index.get(word, UNK_TOKEN) for word in sentence.split(' ')]

def tensor_from_sentence(lang, sentence, max_len, device):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_TOKEN)
    while len(indexes) < max_len:
        indexes.append(PAD_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


# --- 4. Seq2Seq Model with Attention ---

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=False)

    def forward(self, input, hidden):
        output, hidden = self.gru(self.embedding(input).view(1, 1, -1), hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super(LuongAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, decoder_hidden, encoder_outputs):
        attn_scores = torch.sum(decoder_hidden.squeeze(0) * self.attn(encoder_outputs), dim=1)
        return F.softmax(attn_scores, dim=0).unsqueeze(0).unsqueeze(0)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_TOKEN)
        self.attention = LuongAttention(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, decoder_input, decoder_hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(decoder_input).view(1, 1, -1))
        attn_weights = self.attention(decoder_hidden, encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs.unsqueeze(0))
        output, hidden = self.gru(torch.cat((embedded, context), 2), decoder_hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

# --- 5. Training & Evaluation ---

def train_step(input_tensor, target_tensor, encoder, decoder, optimizers, criterion, max_length, device):
    encoder_hidden = encoder.initHidden(device)
    optimizers['encoder'].zero_grad()
    optimizers['decoder'].zero_grad()
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0

    for ei in range(input_tensor.size(0)):
        if input_tensor[ei].item() == PAD_TOKEN: break
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
    decoder_hidden = encoder_hidden

    for di in range(target_tensor.size(0)):
        decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]
        if decoder_input.item() == EOS_TOKEN: break
            
    loss.backward()
    optimizers['encoder'].step()
    optimizers['decoder'].step()
    return loss.item() / target_tensor.size(0)

def run_training(encoder, decoder, pairs, input_lang, output_lang, max_len, device, args):
    optimizers = {'encoder': optim.SGD(encoder.parameters(), lr=args.lr), 'decoder': optim.SGD(decoder.parameters(), lr=args.lr)}
    criterion = nn.NLLLoss(ignore_index=PAD_TOKEN)
    total_loss = 0
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        pair = random.choice(pairs)
        input_tensor = tensor_from_sentence(input_lang, pair[0], max_len, device)
        target_tensor = tensor_from_sentence(output_lang, pair[1], max_len, device)
        loss = train_step(input_tensor, target_tensor, encoder, decoder, optimizers, criterion, max_len, device)
        total_loss += loss
        if epoch % args.print_every == 0:
            print(f"Epoch {epoch}/{args.epochs}, Loss: {total_loss / args.print_every:.4f}")
            total_loss = 0
    print("Training complete.")

def run_translation(encoder, decoder, text, input_lang, output_lang, max_length, device):
    with torch.no_grad():
        input_words = text.split(' ')
        input_tensor = tensor_from_sentence(input_lang, text, max_length, device)
        encoder_hidden = encoder.initHidden(device)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_tensor.size(0)):
            if input_tensor[ei].item() == PAD_TOKEN: break
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
        decoder_hidden = encoder_hidden
        decoded_words = []

        for _ in range(max_length):
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.data.topk(1)
            if topi.item() == UNK_TOKEN:
                _, max_attn_index = attn_weights.max(2)
                decoded_words.append(input_words[max_attn_index.item()] if max_attn_index.item() < len(input_words) else "<unk>")
            elif topi.item() == EOS_TOKEN: break
            else: decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.detach()
        return ' '.join(decoded_words)


# --- 6. Checkpointing ---

def write_checkpoint(encoder, decoder, input_lang, output_lang, max_len, source, target):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE_PATTERN.format(src=source, tar=target))
    torch.save({'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict(),
                'input_lang': input_lang, 'output_lang': output_lang, 'max_len': max_len}, path)
    print(f"Checkpoint saved to {path}")

def read_checkpoint(source, target, device, model_path=None):
    path = model_path or os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE_PATTERN.format(src=source, tar=target))
    if not os.path.exists(path): return None, None
    with safe_globals([Lang]):
        checkpoint = torch.load(path, map_location=device)
    print(f"Loaded checkpoint from {path}")
    return checkpoint, path


# --- 7. CLI Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Zolai-Centric Multilingual NMT CLI Tool", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Define parsers
    train_parser = subparsers.add_parser('train', help='Train a new model')
    translate_parser = subparsers.add_parser('translate', help='Translate a text')
    template_test_parser = subparsers.add_parser('test-template', help='Test the template generation engine')

    # Add arguments to all parsers that need them
    for sub_parser in [train_parser, translate_parser, template_test_parser]:
        sub_parser.add_argument('--source', type=str, required=True, choices=SUPPORTED_LANGUAGES)
        sub_parser.add_argument('--target', type=str, required=True, choices=SUPPORTED_LANGUAGES)

    # Train-specific arguments
    train_parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    train_parser.add_argument('--print-every', type=int, default=CONFIG['print_every'])
    train_parser.add_argument('--lr', type=float, default=CONFIG['learning_rate'])
    train_parser.add_argument('--vocab-size', type=int, default=CONFIG['vocab_size'])
    train_parser.add_argument('--min-count', type=int, default=CONFIG['min_count'])
    train_parser.add_argument('--use-templates', action='store_true', help='Generate training data from YAML templates.')

    # Translate-specific arguments
    translate_parser.add_argument('--text', type=str, required=True)
    translate_parser.add_argument('--model-path', type=str, default=None)

    args = parser.parse_args()

    # --- Validate Arguments ---
    if args.source == args.target: sys.exit("Error: Source and target languages cannot be the same.")
    if PRIMARY_LANGUAGE not in [args.source, args.target]: sys.exit(f"Error: Zolai ('{PRIMARY_LANGUAGE}') must be the source or target.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        if args.command == 'train':
            input_lang, output_lang, pairs, max_len = prepare_data(args.source, args.target, args.vocab_size, args.min_count, args.use_templates)
            print(f"Training {input_lang.name.upper()} -> {output_lang.name.upper()} with {len(pairs)} pairs.")
            encoder = EncoderRNN(input_lang.n_words, CONFIG['hidden_size']).to(device)
            decoder = AttnDecoderRNN(CONFIG['hidden_size'], output_lang.n_words).to(device)
            run_training(encoder, decoder, pairs, input_lang, output_lang, max_len, device, args)
            write_checkpoint(encoder, decoder, input_lang, output_lang, max_len, args.source, args.target)

        elif args.command == 'translate':
            checkpoint, _ = read_checkpoint(args.source, args.target, device, args.model_path)
            if not checkpoint: sys.exit(f"Error: No checkpoint found for '{args.source}-{args.target}'.")
            
            input_lang, output_lang, max_len = checkpoint['input_lang'], checkpoint['output_lang'], checkpoint['max_len']
            encoder = EncoderRNN(input_lang.n_words, CONFIG['hidden_size']).to(device)
            decoder = AttnDecoderRNN(CONFIG['hidden_size'], output_lang.n_words).to(device)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            
            translation = run_translation(encoder, decoder, args.text, input_lang, output_lang, max_len, device)
            print(f"\nSource ({args.source}): '{args.text}'\nTranslation ({args.target}): '{translation}'")

        elif args.command == 'test-template':
            print("Testing template generation...")
            template_pairs = load_templated_data(args.source, args.target)
            if not template_pairs:
                print("No data generated. Check template files and language keys.")
                return

            os.makedirs(TMP_DIR, exist_ok=True)
            output_filename = f"generated_data_{args.source}-{args.target}.tsv"
            output_path = os.path.join(TMP_DIR, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for pair in template_pairs:
                    f.write(f"{pair[0]}\t{pair[1]}\n")
            
            print(f"Successfully generated {len(template_pairs)} pairs to '{output_path}'")


    except (FileNotFoundError, ValueError, AttributeError, RuntimeError) as e:
        sys.exit(f"Error: {e}")

if __name__ == '__main__':
    main()
