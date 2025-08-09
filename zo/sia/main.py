"""
Zolai-Centric Multilingual NMT Framework
version: 2025.08.08.1530

A command-line framework for training, evaluating, and deploying
state-of-the-art Transformer-based Neural Machine Translation models, with a
focus on the Zolai language.

--- Features ---

- Checksum Validation: Automatically detects if training data has changed since
  the tokenizer was last trained, preventing stale vocabulary issues.
- Resumable Training: Safely interrupt and resume from the last checkpoint.
- Centralized Configuration: Manages settings via YAML files with .env overrides.
- Modular Data Preprocessing: All data loading is handled by the `preprocess` module.
- Advanced Semantic Templates: Generate high-quality training data from YAML rules.
- Transformer Architecture: Uses multi-head self-attention for quality translation.

--- CLI Usage Examples ---

# 1. Start a new training run using both parallel TSV and template data
python -m zo.sia.main train zo-en --use-parallel --use-template

# 2. Resume an interrupted training run
python -m zo.sia.main train zo-en --resume
"""
import os
import argparse
import sys
import math
import traceback
import warnings
import hashlib
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.serialization import safe_globals
from tqdm import tqdm

# Local application imports
from zo.sia.config import Config
from zo.sia.preprocess import (
    build_data_index, prepare_tokenizer_data, get_indexed_pairs, 
    load_templated_data, StreamingTranslationDataset, collate_fn
)

# --- 1. Tokenization & Validation ---

class Tokenizer:
    """A wrapper for the SentencePiece tokenizer."""
    def __init__(self, cfg, model_path=None):
        self.cfg = cfg
        self.sp = spm.SentencePieceProcessor()
        if model_path: self.load(model_path)

    def train(self, text_file_path, model_prefix, vocab_size):
        # Use the nested config path: self.cfg.app.tokens
        command = (
            f'--input={text_file_path} --model_prefix={model_prefix} '
            f'--vocab_size={vocab_size} --character_coverage=1.0 '
            f'--model_type=bpe --pad_id={self.cfg.app.tokens.pad} '
            f'--unk_id={self.cfg.app.tokens.unk} --bos_id={self.cfg.app.tokens.sos} '
            f'--eos_id={self.cfg.app.tokens.eos}'
        )
        spm.SentencePieceTrainer.Train(command)
        self.load(f"{model_prefix}.model")

    def load(self, model_path):
        self.sp.Load(model_path)
        self.vocab_size = self.sp.GetPieceSize()

    def encode(self, sentence):
        return self.sp.EncodeAsIds(sentence)

    def decode(self, token_ids):
        return self.sp.DecodeIds(token_ids)

def calculate_file_hash(filepath):
    """Calculates the SHA256 hash of a file's content."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def verify_tokenizer_freshness(cfg, args):
    """
    Verifies that the tokenizer is up-to-date with the training data by
    comparing the hash of the current data with a saved hash.
    """
    source, target = args.lang_pair.split('-')
    lang_pair_sorted = f"{sorted([source, target])[0]}-{sorted([source, target])[1]}"
    hash_filename = cfg.app.filenames.tokenizer_prefix.format(lang_pair=lang_pair_sorted) + ".sha256"
    hash_filepath = os.path.join(cfg.data.paths.experiments_dir, hash_filename)

    if not os.path.exists(hash_filepath):
        sys.exit(f"Error: Tokenizer hash file not found at '{hash_filepath}'.\n"
                 "Please run the 'train-tokenizer' command first to create the tokenizer and its checksum.")

    with open(hash_filepath, 'r') as f:
        saved_hash = f.read().strip()

    print("Verifying data integrity against tokenizer...")
    current_data_file, _ = prepare_tokenizer_data(cfg, source, target, args)
    current_hash = calculate_file_hash(current_data_file)

    if saved_hash != current_hash:
        sys.exit(f"Error: Training data has changed since the tokenizer was last trained.\n"
                 f"The vocabulary is out of date. Please run 'train-tokenizer' for '{args.lang_pair}' again.")
    
    print("Tokenizer is up-to-date. Proceeding with training.")


# --- 2. Transformer Model Architecture ---

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, embed_size, 2) * math.log(10000) / embed_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, embed_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        pos_emb = self.pos_embedding[:token_embedding.size(1), :]
        return self.dropout(token_embedding + pos_emb)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, pad_idx):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.embed_size = embed_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.embed_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, cfg, model_cfg, vocab_size):
        super(Seq2SeqTransformer, self).__init__()
        arch = model_cfg.architecture
        self.transformer = nn.Transformer(
            d_model=arch.embedding_size, nhead=arch.num_heads,
            num_encoder_layers=arch.num_layers, num_decoder_layers=arch.num_layers,
            dim_feedforward=arch.ff_hidden_size, dropout=arch.dropout, batch_first=True
        )
        self.generator = nn.Linear(arch.embedding_size, vocab_size)
        # Use the nested config path: cfg.app.tokens.pad
        self.src_tok_emb = TokenEmbedding(vocab_size, arch.embedding_size, cfg.app.tokens.pad)
        self.tgt_tok_emb = TokenEmbedding(vocab_size, arch.embedding_size, cfg.app.tokens.pad)
        self.positional_encoding = PositionalEncoding(arch.embedding_size, dropout=arch.dropout)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask, src_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(src_emb, src_mask, src_padding_mask)

    def decode(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask, None, tgt_key_padding_mask, memory_key_padding_mask)

def generate_square_subsequent_mask(sz, device):
    return torch.triu(torch.ones((sz, sz), device=device), diagonal=1).bool()

def create_mask(src, tgt, pad_token_id, device):
    src_seq_len, tgt_seq_len = src.shape[1], tgt.shape[1]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = None
    src_padding_mask = (src == pad_token_id)
    tgt_padding_mask = (tgt == pad_token_id)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# --- 3. Training & Evaluation ---

def run_training_loop(model, optimizer, train_loader, val_loader, device, cfg, model_cfg, args, start_epoch, best_val_loss):
    """The main training loop, capable of resuming from a checkpoint."""
    # Use the nested config path: cfg.app.tokens.pad
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.app.tokens.pad)
    patience_counter = 0
    training_params = model_cfg.training

    with tqdm(range(start_epoch, training_params.epochs + 1), initial=start_epoch, total=training_params.epochs, desc="Training Progress") as epoch_iterator:
        for epoch in epoch_iterator:
            model.train()
            train_loss = 0
            for src, tgt in train_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                # Use the nested config path: cfg.app.tokens.pad
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, cfg.app.tokens.pad, device)
                logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                optimizer.zero_grad()
                tgt_out = tgt[:, 1:]
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for src, tgt in val_loader:
                    src, tgt = src.to(device), tgt.to(device)
                    tgt_input = tgt[:, :-1]
                    # Use the nested config path: cfg.app.tokens.pad
                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, cfg.app.tokens.pad, device)
                    logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                    tgt_out = tgt[:, 1:]
                    loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                write_checkpoint(epoch, model, optimizer, best_val_loss, cfg, args)
                epoch_iterator.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}", best_val=f"{best_val_loss:.4f} (saved)")
            else:
                patience_counter += 1
                epoch_iterator.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}", best_val=f"{best_val_loss:.4f}")
                if patience_counter >= training_params.patience: 
                    tqdm.write("\nEarly stopping triggered.")
                    break
    tqdm.write("Training complete.")

def run_translation(model, text, tokenizer, device, cfg):
    """Translates a single text using greedy decoding."""
    model.eval()
    with torch.no_grad():
        # Use the nested config path: cfg.app.tokens
        src_ids = [cfg.app.tokens.sos] + tokenizer.encode(text) + [cfg.app.tokens.eos]
        src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
        src_mask = None
        src_padding_mask = (src_tensor == cfg.app.tokens.pad)
        memory = model.encode(src_tensor, src_mask, src_padding_mask)
        ys = torch.ones(1, 1).fill_(cfg.app.tokens.sos).type(torch.long).to(device)
        for _ in range(50): # Max output length
            tgt_mask = generate_square_subsequent_mask(ys.size(1), device)
            out = model.decode(ys, memory, tgt_mask, None, src_padding_mask)
            prob = model.generator(out[:, -1])
            _, next_word_id = torch.max(prob, dim=1)
            if next_word_id.item() == cfg.app.tokens.eos: break
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word_id.item())], dim=1)
    return tokenizer.decode(ys.squeeze().tolist())

def run_test_evaluation(model, test_pairs, tokenizer, device, cfg, args):
    """Evaluates the model on a test set and saves the results."""
    print(f"Testing model on {len(test_pairs)} sentences...")
    results = []
    for src, tgt in tqdm(test_pairs, desc="Evaluating"):
        translation = run_translation(model, src, tokenizer, device, cfg)
        results.append(f"{src}\t{tgt}\t{translation}\n")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(f"{args.lang_pair.split('-')[0]}\t{args.lang_pair.split('-')[1]}\tmodel_translation\n")
        f.writelines(results)
    print(f"Test complete. Results saved to '{args.output_file}'")

# --- 4. Checkpointing & Setup ---

def write_checkpoint(epoch, model, optimizer, best_val_loss, cfg, args):
    """Saves a complete training checkpoint atomically."""
    # Use the nested config path: cfg.app.filenames
    path = os.path.join(cfg.data.paths.experiments_dir, cfg.app.filenames.checkpoint.format(lang_pair=args.lang_pair))
    tmp_path = path + ".tmp"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'cli_args': vars(args)
    }, tmp_path)
    
    os.replace(tmp_path, path)

def read_checkpoint(cfg, lang_pair, device, model_path=None):
    """Loads a complete training checkpoint."""
    # Use the nested config path: cfg.app.filenames
    path = model_path or os.path.join(cfg.data.paths.experiments_dir, cfg.app.filenames.checkpoint.format(lang_pair=lang_pair))
    if not os.path.exists(path): return None
    # Add safe_globals to handle loading argparse.Namespace
    with safe_globals([argparse.Namespace]):
        checkpoint = torch.load(path, map_location=device)
    print(f"Loaded checkpoint from {os.path.basename(path)}")
    return checkpoint

def setup_model_and_optimizer(cfg, model_cfg, tokenizer, device):
    """Initializes the model and optimizer based on configuration."""
    model = Seq2SeqTransformer(cfg, model_cfg, tokenizer.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=model_cfg.training.learning_rate)
    return model, optimizer

# --- 5. CLI Command Handlers ---

def handle_train(cfg, args, device):
    """Handles the 'train' command."""
    # First, verify the tokenizer is up-to-date
    verify_tokenizer_freshness(cfg, args)

    source, target = args.lang_pair.split('-')
    lang_pair_sorted = sorted([source, target])
    # Use the nested config path: cfg.app.filenames
    tokenizer_path = os.path.join(cfg.data.paths.experiments_dir, cfg.app.filenames.tokenizer_prefix.format(lang_pair=f"{lang_pair_sorted[0]}-{lang_pair_sorted[1]}") + ".model")
    
    tokenizer = Tokenizer(cfg, tokenizer_path)
    model_cfg = cfg.get_lang_pair_config(source, target)
    model, optimizer = setup_model_and_optimizer(cfg, model_cfg, tokenizer, device)
    
    start_epoch = 1
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = read_checkpoint(cfg, args.lang_pair, device)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("Warning: --resume flag was used, but no checkpoint was found. Starting new training.")

    data_index = build_data_index(cfg, source, target, args)
    split_idx = int(len(data_index) * (1 - model_cfg.training.validation_split))
    train_dataset = StreamingTranslationDataset(data_index[:split_idx], tokenizer, cfg)
    val_dataset = StreamingTranslationDataset(data_index[split_idx:], tokenizer, cfg)
    
    # Use the nested config path: cfg.app.tokens.pad
    collate_with_pad = lambda batch: collate_fn(batch, cfg.app.tokens.pad)
    
    train_loader = DataLoader(train_dataset, batch_size=model_cfg.training.batch_size, shuffle=True, collate_fn=collate_with_pad)
    val_loader = DataLoader(val_dataset, batch_size=model_cfg.training.batch_size, shuffle=False, collate_fn=collate_with_pad)
    
    run_training_loop(model, optimizer, train_loader, val_loader, device, cfg, model_cfg, args, start_epoch, best_val_loss)

def handle_translate(cfg, args, device):
    """Handles the 'translate' command."""
    source, target = args.lang_pair.split('-')
    lang_pair_sorted = sorted([source, target])
    # Use the nested config path: cfg.app.filenames
    tokenizer_path = os.path.join(cfg.data.paths.experiments_dir, cfg.app.filenames.tokenizer_prefix.format(lang_pair=f"{lang_pair_sorted[0]}-{lang_pair_sorted[1]}") + ".model")
    tokenizer = Tokenizer(cfg, tokenizer_path)

    checkpoint = read_checkpoint(cfg, args.lang_pair, device, args.model_path)
    if not checkpoint: sys.exit(f"Error: No checkpoint found for '{args.lang_pair}'.")
    
    model_cfg = cfg.get_lang_pair_config(source, target)
    model, _ = setup_model_and_optimizer(cfg, model_cfg, tokenizer, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    translation = run_translation(model, args.text, tokenizer, device, cfg)
    print(f"\nSource ({source}): '{args.text}'\nTranslation ({target}): '{translation}'")

def handle_test(cfg, args, device):
    """Handles the 'test' command for model evaluation."""
    source, target = args.lang_pair.split('-')
    lang_pair_sorted = sorted([source, target])
    # Use the nested config path: cfg.app.filenames
    tokenizer_path = os.path.join(cfg.data.paths.experiments_dir, cfg.app.filenames.tokenizer_prefix.format(lang_pair=f"{lang_pair_sorted[0]}-{lang_pair_sorted[1]}") + ".model")
    tokenizer = Tokenizer(cfg, tokenizer_path)

    checkpoint = read_checkpoint(cfg, args.lang_pair, device, args.model_path)
    if not checkpoint: sys.exit(f"Error: No checkpoint found for '{args.lang_pair}'.")
    
    model_cfg = cfg.get_lang_pair_config(source, target)
    model, _ = setup_model_and_optimizer(cfg, model_cfg, tokenizer, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_pairs = get_indexed_pairs(cfg, source, target, 'test')
    if not test_pairs: sys.exit(f"Error: No test data found for '{args.lang_pair}' in datasets.yaml.")
    
    run_test_evaluation(model, test_pairs, tokenizer, device, cfg, args)

def handle_train_tokenizer(cfg, args):
    """Handles the 'train-tokenizer' command."""
    source, target = args.lang_pair.split('-')
    text_file, unique_words = prepare_tokenizer_data(cfg, source, target, args)
    
    model_cfg = cfg.get_lang_pair_config(source, target)
    final_vocab_size = min(model_cfg.tokenizer.vocab_size, unique_words)
    if final_vocab_size < model_cfg.tokenizer.vocab_size:
        print(f"Warning: Capping vocab size at {unique_words} (the number of unique words found).")
    
    print(f"Training tokenizer on {os.path.basename(text_file)} with vocab size {final_vocab_size}...")
    lang_pair_sorted = sorted([source, target])
    lang_pair_str = f"{lang_pair_sorted[0]}-{lang_pair_sorted[1]}"
    # Use the nested config path: cfg.app.filenames
    model_prefix = os.path.join(cfg.data.paths.experiments_dir, cfg.app.filenames.tokenizer_prefix.format(lang_pair=lang_pair_str))
    
    Tokenizer(cfg).train(text_file, model_prefix, final_vocab_size)
    print(f"Tokenizer model saved to {model_prefix}.model")

    # Calculate and save the hash of the data used for training
    data_hash = calculate_file_hash(text_file)
    hash_filename = cfg.app.filenames.tokenizer_prefix.format(lang_pair=lang_pair_str) + ".sha256"
    hash_filepath = os.path.join(cfg.data.paths.experiments_dir, hash_filename)
    with open(hash_filepath, 'w') as f:
        f.write(data_hash)
    print(f"Data fingerprint saved to {hash_filename}")


def handle_test_data_generation(cfg, args, source_func):
    """Generic handler for testing data generation (parallel or template)."""
    source, target = args.lang_pair.split('-')
    pairs = source_func(cfg, source, target, args)
    if not pairs:
        print("No data generated.")
        return
        
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(f"{source}\t{target}\n")
        for pair in pairs:
            f.write(f"{pair[0]}\t{pair[1]}\n")
    print(f"Successfully generated {len(pairs)} pairs to '{args.output_file}'")

# --- 6. Main Execution ---

def main():
    # --- Argument Parsing Setup ---
    parser = argparse.ArgumentParser(description="Zolai-Centric Multilingual NMT CLI Tool", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('lang_pair', type=str, help="Language pair, e.g., 'zo-en'")
    
    # --- Tag Filtering Arguments ---
    tag_parser = argparse.ArgumentParser(add_help=False)
    tag_parser.add_argument('--include-tags', type=str, help="Comma-separated list of tags to include for template generation.")
    tag_parser.add_argument('--exclude-tags', type=str, help="Comma-separated list of tags to exclude for template generation.")

    # --- Data Source Arguments ---
    data_source_parser = argparse.ArgumentParser(add_help=False)
    data_source_parser.add_argument('--use-parallel', action='store_true', help="Include data from parallel TSV files listed in datasets.yaml.")
    data_source_parser.add_argument('--use-template', action='store_true', help="Include data generated from YAML templates.")

    # --- Command Parsers ---
    train_parser = subparsers.add_parser('train', parents=[parent_parser, tag_parser, data_source_parser], help='Train a new model or resume training.')
    train_parser.add_argument('--resume', action='store_true', help="Resume training from the last checkpoint.")

    translate_parser = subparsers.add_parser('translate', parents=[parent_parser], help='Translate a single sentence.')
    translate_parser.add_argument('--text', type=str, required=True, help="The sentence to translate.")
    translate_parser.add_argument('--model-path', type=str, help="Optional path to a specific checkpoint file.")

    test_parser = subparsers.add_parser('test', parents=[parent_parser], help='Evaluate the model on the official test set.')
    test_parser.add_argument('--output-file', type=str, required=True, help="Path to save the TSV results file.")
    test_parser.add_argument('--model-path', type=str, help="Optional path to a specific checkpoint file.")

    tokenizer_parser = subparsers.add_parser('train-tokenizer', parents=[parent_parser, data_source_parser, tag_parser], help='Train the SentencePiece tokenizer.')

    test_parallel_parser = subparsers.add_parser('test-parallel', parents=[parent_parser], help='Test the parallel TSV data loader.')
    test_parallel_parser.add_argument('--output-file', type=str, required=True, help="Path to save the generated data.")
    
    test_template_parser = subparsers.add_parser('test-template', parents=[parent_parser, tag_parser], help='Test the template generation engine.')
    test_template_parser.add_argument('--output-file', type=str, required=True, help="Path to save the generated data.")

    args = parser.parse_args()

    # --- Main Logic ---
    try:
        cfg = Config()
        
        # Validate lang_pair format and languages using the nested config
        try:
            source, target = args.lang_pair.split('-')
            if source == target: sys.exit("Error: Source and target languages cannot be the same.")
            if source not in cfg.app.supported_languages or target not in cfg.app.supported_languages:
                sys.exit(f"Error: Unsupported language in pair '{args.lang_pair}'. Supported: {cfg.app.supported_languages}")
            if cfg.app.primary_language not in [source, target]:
                sys.exit(f"Error: Zolai ('{cfg.app.primary_language}') must be the source or target.")
        except ValueError:
            sys.exit(f"Error: Invalid lang_pair format '{args.lang_pair}'. Use 'source-target', e.g., 'zo-en'.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Suppress the specific "nested tensors" prototype warning from PyTorch.
        warnings.filterwarnings("ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning)

        # Create necessary directories using the correct config paths
        os.makedirs(cfg.data.paths.experiments_dir, exist_ok=True)
        os.makedirs(cfg.data.paths.tmp_dir, exist_ok=True)

        # Route to the correct handler based on the command
        if args.command == 'train': handle_train(cfg, args, device)
        elif args.command == 'translate': handle_translate(cfg, args, device)
        elif args.command == 'test': handle_test(cfg, args, device)
        elif args.command == 'train-tokenizer': handle_train_tokenizer(cfg, args)
        elif args.command == 'test-parallel': handle_test_data_generation(cfg, args, lambda c, s, t, a: get_indexed_pairs(c, s, t))
        elif args.command == 'test-template': handle_test_data_generation(cfg, args, load_templated_data)

    except (FileNotFoundError, ValueError, AttributeError, RuntimeError, KeyError) as e:
        print(f"\nAn unexpected error occurred: {type(e).__name__}: {e}", file=sys.stderr)
        print("-------------------", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
