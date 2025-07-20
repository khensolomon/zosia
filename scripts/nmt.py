# v.0004
"""
Zolai/English Bidirectional Neural Machine Translation CLI Tool

This script provides a command-line interface for training a basic sequence-to-sequence
Neural Machine Translation (NMT) model and using it to translate sentences between
Zolai (zo) and English (en).

Features:
- Bidirectional training (zo -> en and en -> zo).
- Loads data from specified TSV files.
- Simple vocabulary management.
- Basic Encoder-Decoder RNN architecture.
- Checkpointing to save and load model progress.
- CLI for easy interaction.

Usage examples:
  # Train a model for Zolai to English translation
  python ./scripts/nmt.py train --mode zo2en --epochs 10000

  # Train a model for English to Zolai translation
  python ./scripts/nmt.py train --mode en2zo --epochs 10000

  # Translate a Zolai sentence to English using the latest checkpoint
  python ./scripts/nmt.py translate --mode zo2en --sentence "kei hong paita"

  # Translate an English sentence to Zolai
  python ./scripts/nmt.py translate --mode en2zo --sentence "how are you"
"""
import os
import io
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Configuration and Constants ---

# Special tokens
SOS_TOKEN = 0  # Start Of Sequence
EOS_TOKEN = 1  # End Of Sequence
PAD_TOKEN = 2  # Padding token

# File and directory paths
CHECKPOINT_DIR = "./experiments"
DATA_FILES = {
    "zo2en": ["./data/corpus/zo-en.tsv"],
    "en2zo": ["./data/corpus/en-zo.tsv"]
}

# Model and Training Parameters
CONFIG = {
    'hidden_size': 256,
    'learning_rate': 0.01,
    'epochs': 75000,
    'print_every': 5000,
    'default_mode': 'zo2en',
}


# --- 2. Data Preparation and Vocabulary ---

class Lang:
    """A helper class to manage the vocabulary for a language."""
    def __init__(self, name):
        self.name = name
        self.word2index = {"<SOS>": SOS_TOKEN, "<EOS>": EOS_TOKEN, "<PAD>": PAD_TOKEN}
        self.word2count = {}
        self.index2word = {SOS_TOKEN: "<SOS>", EOS_TOKEN: "<EOS>", PAD_TOKEN: "<PAD>"}
        self.n_words = 3  # Count SOS, EOS, PAD

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def read_data(file_paths):
    """Reads sentence pairs from a list of TSV files."""
    pairs = []
    for file_path in file_paths:
        try:
            with io.open(file_path, encoding='utf-8') as f:
                for line in f:
                    # Skip header or empty lines
                    if '\t' not in line or "sentence" in line:
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        pairs.append(parts)
        except FileNotFoundError:
            print(f"Warning: Data file not found: {file_path}")
    return pairs

def prepare_data(mode):
    """Prepares language objects and sentence pairs for a given mode."""
    file_paths = DATA_FILES.get(mode)
    if not file_paths:
        raise ValueError(f"Invalid mode specified: {mode}. Must be one of {list(DATA_FILES.keys())}")

    pairs = read_data(file_paths)
    if not pairs:
        raise ValueError(f"No data found for mode '{mode}'. Check your data files.")

    # In zo2en mode, input is Zolai, output is English.
    # In en2zo mode, input is English, output is Zolai.
    if mode == 'en2zo':
        pairs = [list(reversed(p)) for p in pairs]

    input_lang = Lang("zolai" if mode == 'zo2en' else "english")
    output_lang = Lang("english" if mode == 'zo2en' else "zolai")

    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    max_len = 0
    if pairs:
        # FIX: More efficient calculation of max sentence length across both languages
        max_len = max(len(s.split(' ')) for p in pairs for s in p)
    # Add buffer for SOS/EOS tokens
    max_len += 2

    return input_lang, output_lang, pairs, max_len

def indexes_from_sentence(lang, sentence):
    # Return index for word if it exists, otherwise return PAD_TOKEN for unknown words
    return [lang.word2index.get(word, PAD_TOKEN) for word in sentence.split(' ')]

def tensor_from_sentence(lang, sentence, max_len, device):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_TOKEN)
    # Pad sequence
    while len(indexes) < max_len:
        indexes.append(PAD_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


# --- 3. Seq2Seq Model Definition ---

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# --- 4. Training ---

def train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, device):
    encoder_hidden = encoder.initHidden(device)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    # --- Encoder ---
    for ei in range(input_tensor.size(0)):
        # Stop encoding if we hit padding
        if input_tensor[ei].item() == PAD_TOKEN:
            break
        _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    # --- Decoder ---
    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
    decoder_hidden = encoder_hidden

    for di in range(target_tensor.size(0)):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_tensor[di])
        
        # Teacher forcing
        decoder_input = target_tensor[di].unsqueeze(0)
        if decoder_input.item() == EOS_TOKEN:
            break
            
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_tensor.size(0)

def run_training(encoder, decoder, pairs, input_lang, output_lang, max_len, device, n_epochs, print_every, learning_rate):
    encoder.train()
    decoder.train()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_TOKEN)
    
    total_loss = 0
    for epoch in range(1, n_epochs + 1):
        pair = random.choice(pairs)
        input_tensor = tensor_from_sentence(input_lang, pair[0], max_len, device)
        target_tensor = tensor_from_sentence(output_lang, pair[1], max_len, device)

        loss = train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_len, device)
        total_loss += loss

        if epoch % print_every == 0:
            avg_loss = total_loss / print_every
            print(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.4f}")
            total_loss = 0


# --- 5. Evaluation (Translation) ---

def run_translation(encoder, decoder, sentence, input_lang, output_lang, max_length, device):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence, max_length, device)
        encoder_hidden = encoder.initHidden(device)

        for ei in range(input_tensor.size(0)):
            if input_tensor[ei].item() == PAD_TOKEN:
                break
            _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
        decoder_hidden = encoder_hidden
        decoded_words = []

        for _ in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            
            if topi.item() == EOS_TOKEN or topi.item() == PAD_TOKEN:
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            # FIX: Do not use .squeeze() here. The decoder input must maintain its dimensions.
            decoder_input = topi.detach()

        return ' '.join(decoded_words)


# --- 6. Checkpointing ---

def save_checkpoint(encoder, decoder, mode, epoch):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filename = f"checkpoint_{mode}_{epoch}.pt"
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")

def get_latest_checkpoint_path(mode):
    """Finds the latest checkpoint file for a given mode."""
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith(f"checkpoint_{mode}_") and f.endswith(".pt")]
    if not files:
        return None
        
    # Sort by epoch number to get the latest
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    return os.path.join(CHECKPOINT_DIR, files[0])

def load_checkpoint(path, encoder, decoder):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    # Load onto the same device the model will be on
    map_location = next(encoder.parameters()).device
    checkpoint = torch.load(path, map_location=map_location)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    print(f"Loaded checkpoint from {path}")


# --- 7. CLI Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Zolai-English NMT CLI Tool")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- Train Command ---
    parser_train = subparsers.add_parser('train', help='Train a new model')
    parser_train.add_argument('--mode', type=str, choices=DATA_FILES.keys(), default=CONFIG['default_mode'], help='Translation direction (e.g., zo2en)')
    parser_train.add_argument('--epochs', type=int, default=CONFIG['epochs'], help='Number of training iterations')
    parser_train.add_argument('--print-every', type=int, default=CONFIG['print_every'], help='How often to print loss')
    parser_train.add_argument('--lr', type=float, default=CONFIG['learning_rate'], help='Learning rate')

    # --- Translate Command ---
    parser_translate = subparsers.add_parser('translate', help='Translate a sentence')
    parser_translate.add_argument('--mode', type=str, choices=DATA_FILES.keys(), default=CONFIG['default_mode'], help='Translation direction')
    parser_translate.add_argument('--sentence', type=str, required=True, help='The sentence to translate')
    parser_translate.add_argument('--model-path', type=str, default=None, help='Path to a specific model checkpoint. If not provided, uses the latest.')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data for the selected mode
    input_lang, output_lang, pairs, max_len = prepare_data(args.mode)
    print(f"Mode: {args.mode.upper()}")
    print(f"Input Lang ({input_lang.name}): {input_lang.n_words} words")
    print(f"Output Lang ({output_lang.name}): {output_lang.n_words} words")
    print(f"Max sentence length: {max_len}")

    # Initialize models
    encoder = EncoderRNN(input_lang.n_words, CONFIG['hidden_size']).to(device)
    decoder = DecoderRNN(CONFIG['hidden_size'], output_lang.n_words).to(device)

    if args.command == 'train':
        print("Starting training...")
        run_training(encoder, decoder, pairs, input_lang, output_lang, max_len, device,
                     n_epochs=args.epochs, print_every=args.print_every, learning_rate=args.lr)
        # Save final model
        save_checkpoint(encoder, decoder, args.mode, args.epochs)
        print("Training complete.")

    elif args.command == 'translate':
        checkpoint_path = args.model_path or get_latest_checkpoint_path(args.mode)
        if not checkpoint_path:
            print(f"Error: No checkpoint found for mode '{args.mode}'. Please train a model first.")
            return

        load_checkpoint(checkpoint_path, encoder, decoder)
        translation = run_translation(encoder, decoder, args.sentence, input_lang, output_lang, max_len, device)
        print(f"\nSource: '{args.sentence}'")
        print(f"Translation: '{translation}'")


if __name__ == '__main__':
    main()
