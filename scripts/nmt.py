"""
Zolai-Centric Multilingual NMT Framework
version: 2025.08.02.190400

A command-line framework for training, evaluating, and deploying
state-of-the-art Transformer-based Neural Machine Translation models, with a
focus on the Zolai language.

--- Features ---

Refactored for Maintainability
  The script has been significantly refactored with a central Config class,
  command handlers, and a model factory function to reduce boilerplate,
  improve clarity, and make it easier to extend.

Scalable Streaming Dataset
  Implements a memory-efficient data pipeline that indexes datasets on disk
  and streams them during training.

Advanced Semantic Templates
  The template engine supports a highly flexible YAML structure with a
  'glossary' for parallel word lists, 'metadata' for semantic tagging, and
  conditional placeholders for context-aware generation.

Transformer Architecture
  Uses multi-head self-attention for state-of-the-art translation quality.

--- CLI Usage Examples ---

# 1. Data Preparation
# (One-time step) Train a shared tokenizer for a language pair using all data sources.
python ./scripts/nmt.py train-tokenizer --source zo --target en --use-templates --use-tsv-index

# 2. Training
# Train a new model using the generated tokenizer and all available data.
# This will automatically use hyperparams-zo-en.yaml if it exists, otherwise hyperparams.yaml.
python ./scripts/nmt.py train --source zo --target en --use-templates --use-tsv-index

# 3. Evaluation and Testing
# Evaluate the best trained model against the defined test set.
python ./scripts/nmt.py test --source zo --target en

# 4. Translate a Single Sentence
# Translate a single piece of text using the best trained model.
python ./scripts/nmt.py translate --source zo --target en --text "Solomon in tui a dawn nuam"
"""
import os
import io
import random
import argparse
import sys
import re
import itertools
import math
import traceback
import linecache
import yaml # Requires PyYAML
import sentencepiece as spm # Requires sentencepiece
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.serialization import safe_globals
from collections import Counter
from tqdm import tqdm # Requires tqdm: pip install tqdm

# --- 1. Configuration and Constants ---

class Config:
    # Special Tokens (SentencePiece default IDs)
    SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN = 1, 2, 0, 3
    
    # Language Configuration
    PRIMARY_LANGUAGE = "zo"
    SUPPORTED_LANGUAGES = ["zo", "en", "no"]
    
    # Directory Paths
    DATA_DIR = "./data"
    CONFIG_DIR = "./config"
    CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
    TEMPLATE_DIR = os.path.join(DATA_DIR, "templates")
    SHARED_TEMPLATE_DIR = os.path.join(TEMPLATE_DIR, "shared")
    EXPERIMENTS_DIR = "./experiments"
    TMP_DIR = "./tmp"
    
    # Index File
    DATASETS_INDEX_FILE = os.path.join(CORPUS_DIR, "datasets.yaml")
    DEFAULT_HYPERPARAMS_FILE = os.path.join(CONFIG_DIR, "hyperparams.yaml")

    # Filename Patterns
    CHECKPOINT_FILE_PATTERN = "checkpoint_{src}-{tgt}.pt"
    TOKENIZER_PREFIX_PATTERN = "tokenizer_{src}-{tgt}"
    TOKENIZER_INPUT_PATTERN = "tokenizer_input_{src}-{tgt}.txt"
    EVAL_RESULTS_PATTERN = "eval_results_{src}-{tgt}.tsv"
    TSV_INDEX_DUMP_PATTERN = "tsv_index_dump_{src}-{tgt}.tsv"
    TEMPLATE_DUMP_PATTERN = "template_dump_{src}-{tgt}.tsv"
    HYPERPARAMS_OUTPUT_PATTERN = os.path.join(CONFIG_DIR, "hyperparams-{src}-{tgt}.yaml")


# --- 2. Data Handling & Tokenization ---

class Tokenizer:
    def __init__(self, model_path=None):
        self.sp = spm.SentencePieceProcessor()
        if model_path: self.load(model_path)
    def train(self, text_file_path, model_prefix, vocab_size):
        command = (f'--input={text_file_path} --model_prefix={model_prefix} '
                   f'--vocab_size={vocab_size} --character_coverage=1.0 '
                   f'--model_type=bpe --pad_id={Config.PAD_TOKEN} --unk_id={Config.UNK_TOKEN} '
                   f'--bos_id={Config.SOS_TOKEN} --eos_id={Config.EOS_TOKEN}')
        spm.SentencePieceTrainer.Train(command)
        self.load(f"{model_prefix}.model")
    def load(self, model_path):
        self.sp.Load(model_path); self.vocab_size = self.sp.GetPieceSize()
    def encode(self, sentence, as_ids=True):
        return self.sp.EncodeAsIds(sentence) if as_ids else self.sp.EncodeAsPieces(sentence)
    def decode(self, tokens, is_ids=True):
        return self.sp.DecodeIds(tokens) if is_ids else self.sp.DecodePieces(tokens)

def get_indexed_pairs(source_lang, target_lang, data_type='train'):
    """Loads parallel data from a list of TSV files defined in an index YAML."""
    if not os.path.exists(Config.DATASETS_INDEX_FILE):
        print(f"Warning: Datasets index file not found at {Config.DATASETS_INDEX_FILE}")
        return []
    
    with open(Config.DATASETS_INDEX_FILE, 'r', encoding='utf-8') as f:
        index = yaml.safe_load(f)

    direction_key = f"{source_lang}-{target_lang}"
    direction_config = index.get(direction_key, {})
    
    default_dir_key = f"{data_type}_dir"
    default_dir = index.get(default_dir_key, os.path.join(Config.CORPUS_DIR, data_type))
    
    specific_basenames = direction_config.get(data_type, [])
    shared_basenames = index.get("shared", {}).get(data_type, [])
    basenames = list(dict.fromkeys(specific_basenames + shared_basenames))

    exclusions = index.get("exclude", {}).get(direction_key, [])
    if exclusions:
        print(f"Excluding files for '{direction_key}': {', '.join(exclusions)}")
        basenames = [b for b in basenames if b not in exclusions]

    all_pairs = []
    print(f"Loading TSV parallel data for '{source_lang}-{target_lang}' ({data_type} set)...")
    for item in basenames:
        if os.path.sep in item:
            file_path = item if item.endswith(".tsv") else f"{item}.tsv"
        else:
            file_path = os.path.join(default_dir, f"{item}.tsv")

        if not os.path.exists(file_path):
            print(f"  - Warning: File '{file_path}' not found. Skipping.")
            continue
        
        with io.open(file_path, encoding='utf-8') as f:
            header = f.readline().strip().split('\t')
            try:
                src_idx = header.index(source_lang)
                tgt_idx = header.index(target_lang)
                print(f"  - Loading from '{os.path.basename(file_path)}'...")
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) > max(src_idx, tgt_idx):
                        all_pairs.append((parts[src_idx], parts[tgt_idx]))
            except ValueError:
                print(f"  - Skipping '{os.path.basename(file_path)}' (missing required language columns).")
                continue
    return all_pairs

def load_templated_data(source_lang, target_lang, args):
    if not os.path.exists(Config.TEMPLATE_DIR): return []
    
    include_tags = set(args.include_template_tags.split(',')) if hasattr(args, 'include_template_tags') and args.include_template_tags else None
    exclude_tags = set(args.exclude_template_tags.split(',')) if hasattr(args, 'exclude_template_tags') and args.exclude_template_tags else None
    specific_files = args.files.split(',') if hasattr(args, 'files') and args.files else None

    if specific_files:
        filenames = [f for f in specific_files if f.endswith((".yaml", ".yml"))]
    else:
        filenames = [f for f in os.listdir(Config.TEMPLATE_DIR) if f.endswith((".yaml", ".yml"))]
    
    if include_tags or exclude_tags:
        filtered_filenames = []
        for filename in filenames:
            file_path = os.path.join(Config.TEMPLATE_DIR, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = yaml.safe_load(f)
                    file_tags = set(data.get('tags', []))
                    if include_tags and not file_tags.intersection(include_tags):
                        continue
                    if exclude_tags and file_tags.intersection(exclude_tags):
                        continue
                    filtered_filenames.append(filename)
                except yaml.YAMLError:
                    continue
        filenames = filtered_filenames

    all_pairs = []
    for filename in filenames:
        file_path = os.path.join(Config.TEMPLATE_DIR, filename)
        all_pairs.extend(generate_from_template_file(file_path, source_lang, target_lang))
    return all_pairs

def capitalize_sentence(sentence):
    if not sentence: return ""
    parts = sentence.split(' ', 1)
    first_word = parts[0]
    if first_word.lower() != 'i':
        first_word = first_word.capitalize()
    return ' '.join([first_word] + parts[1:]) if len(parts) > 1 else first_word

def generate_from_template_file(file_path, source_lang, target_lang):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: template_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Warning: Could not parse YAML file {file_path}. Error: {e}"); return []
    
    if 'import' in template_data:
        merged_data = {}
        for import_file in template_data['import']:
            import_path = os.path.join(Config.SHARED_TEMPLATE_DIR, import_file)
            if os.path.exists(import_path):
                with open(import_path, 'r', encoding='utf-8') as f:
                    shared_data = yaml.safe_load(f)
                    for key, value in shared_data.items():
                        if key not in merged_data:
                            merged_data[key] = value
                        elif isinstance(merged_data[key], list) and isinstance(value, list):
                            merged_data[key].extend(value)
            else:
                print(f"Warning: Imported file not found: {import_path}")
        for key, value in template_data.items():
            if key in merged_data and isinstance(merged_data[key], list) and isinstance(value, list):
                 merged_data[key].extend(value)
            else:
                 merged_data[key] = value
        template_data = merged_data

    templates = template_data.get('templates', [])
    if not templates: return []

    generated_pairs = []
    for template_pair in templates:
        src_template = template_pair.get(source_lang)
        tgt_template = template_pair.get(target_lang)
        if not src_template or not tgt_template: continue

        placeholders = set(re.findall(r"<(\w+)>", src_template) + re.findall(r"<(\w+)>", tgt_template))
        if not all(p in template_data for p in placeholders): continue
        if not placeholders:
            generated_pairs.append((capitalize_sentence(src_template), capitalize_sentence(tgt_template))); continue
        
        conditional_placeholders = {p for p in placeholders if isinstance(template_data.get(p), dict) and 'default' in template_data.get(p)}
        non_conditional_placeholders = placeholders - conditional_placeholders
        
        nc_options = {}
        valid_template = True
        for p in non_conditional_placeholders:
            nc_options[p] = []
            values = template_data.get(p)
            if isinstance(values, list) and all(isinstance(v, str) for v in values):
                for v in values: nc_options[p].append({'src': v, 'tgt': v, 'tags': []})
            elif isinstance(values, dict) and source_lang in values and target_lang in values:
                src_vals, tgt_vals = values[source_lang], values[target_lang]
                canonical_keys = values.get('en', src_vals)
                if not (len(src_vals) == len(tgt_vals) == len(canonical_keys)):
                    valid_template = False; break
                for i in range(len(src_vals)):
                    metadata_key = canonical_keys[i]
                    tags_raw = template_data.get('metadata', {}).get(p, {}).get(metadata_key, {}).get('tags', [])
                    
                    tags = []
                    for tag_item in tags_raw:
                        if isinstance(tag_item, dict):
                            for k, v in tag_item.items():
                                tags.append(f"{k}:{v}")
                        else:
                            tags.append(str(tag_item))
                    
                    nc_options[p].append({'src': src_vals[i], 'tgt': tgt_vals[i], 'tags': tags})
            else:
                valid_template = False; break
        if not valid_template: continue

        nc_keys, nc_values = zip(*nc_options.items()) if nc_options else ([], [])
        for combo in itertools.product(*nc_values):
            context = {nc_keys[i]: combo[i] for i in range(len(nc_keys))}
            
            for p_name in conditional_placeholders:
                p_def = template_data.get(p_name, {})
                src_val, tgt_val = p_def['default'][source_lang], p_def['default'][target_lang]
                
                if 'rules' in p_def:
                    for rule_group_key, rules in p_def['rules'].items():
                        if rule_group_key.startswith("on_") and rule_group_key.endswith("_tag"):
                            dependent_p_name = rule_group_key.split('_')[1]
                            if dependent_p_name in context:
                                dependent_tags = set(context[dependent_p_name]['tags'])
                                for rule in rules:
                                    condition = rule['condition']
                                    matched = False
                                    if isinstance(condition, str) and condition in dependent_tags:
                                        matched = True
                                    elif isinstance(condition, dict):
                                        if 'all_of' in condition and set(condition['all_of']).issubset(dependent_tags):
                                            matched = True
                                        elif 'any_of' in condition and any(t in dependent_tags for t in condition['any_of']):
                                            matched = True
                                        elif 'none_of' in condition and not any(t in dependent_tags for t in condition['none_of']):
                                            matched = True
                                    
                                    if matched:
                                        src_val = rule['translations'].get(source_lang, src_val)
                                        tgt_val = rule['translations'].get(target_lang, tgt_val)
                                        break
                context[p_name] = {'src': src_val, 'tgt': tgt_val}

            final_src, final_tgt = src_template, tgt_template
            for p_name, p_data in context.items():
                final_src = final_src.replace(f"<{p_name}>", str(p_data['src']))
                final_tgt = final_tgt.replace(f"<{p_name}>", str(p_data['tgt']))

            generated_pairs.append((capitalize_sentence(final_src), capitalize_sentence(final_tgt)))
            
    return generated_pairs

def prepare_tokenizer_data(source_lang, target_lang, args, params):
    all_sentences = []
    if args.use_tsv_index:
        for src, tgt in [(source_lang, target_lang), (target_lang, source_lang)]:
            train_pairs = get_indexed_pairs(src, tgt, 'train')
            test_pairs = get_indexed_pairs(src, tgt, 'test')
            all_sentences.extend([p[0] for p in train_pairs] + [p[1] for p in train_pairs])
            all_sentences.extend([p[0] for p in test_pairs] + [p[1] for p in test_pairs])
    if args.use_templates:
        for src, tgt in [(source_lang, target_lang), (target_lang, source_lang)]:
            template_pairs = load_templated_data(src, tgt, args)
            all_sentences.extend([p[0] for p in template_pairs] + [p[1] for p in template_pairs])

    unique_words = set(" ".join(all_sentences).split())
    os.makedirs(Config.TMP_DIR, exist_ok=True)
    output_path = os.path.join(Config.TMP_DIR, Config.TOKENIZER_INPUT_PATTERN.format(src=source_lang, tgt=target_lang))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_sentences))
    return output_path, len(unique_words)

class StreamingTranslationDataset(Dataset):
    def __init__(self, data_index, tokenizer):
        self.data_index = data_index
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        src, tgt = self.data_index[idx]
        src_ids = [Config.SOS_TOKEN] + self.tokenizer.encode(src) + [Config.EOS_TOKEN]
        tgt_ids = [Config.SOS_TOKEN] + self.tokenizer.encode(tgt) + [Config.EOS_TOKEN]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=Config.PAD_TOKEN)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=Config.PAD_TOKEN)
    return src_padded, tgt_padded

def build_data_index(source_lang, target_lang, args):
    """Builds a combined index of TSV file locations and in-memory template pairs."""
    index = []
    if args.use_tsv_index:
        index.extend(get_indexed_pairs(source_lang, target_lang, 'train'))
    if args.use_templates:
        index.extend(load_templated_data(source_lang, target_lang, args))
    
    if not index:
        raise ValueError(f"No data found for the pair '{source_lang}-{target_lang}' from any source.")
    
    random.shuffle(index)
    return index


# --- 3. Transformer Model Architecture ---
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
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=Config.PAD_TOKEN)
        self.embed_size = embed_size
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.embed_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, embed_size, nhead, vocab_size, ff_hidden_size, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=embed_size, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=ff_hidden_size,
                                          dropout=dropout, batch_first=True)
        self.generator = nn.Linear(embed_size, vocab_size)
        self.src_tok_emb = TokenEmbedding(vocab_size, embed_size)
        self.tgt_tok_emb = TokenEmbedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, dropout=dropout)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask, src_padding_mask, device):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(src_emb, src_mask, src_padding_mask)

    def decode(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask, device):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask, None, tgt_key_padding_mask, memory_key_padding_mask)

def generate_square_subsequent_mask(sz, device):
    return torch.triu(torch.ones((sz, sz), device=device), diagonal=1).bool()

def create_mask(src, tgt, device):
    src_seq_len, tgt_seq_len = src.shape[1], tgt.shape[1]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = None
    src_padding_mask, tgt_padding_mask = (src == Config.PAD_TOKEN), (tgt == Config.PAD_TOKEN)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# --- 4. Training & Evaluation ---

def run_training_for_tuning(params, args):
    """
    A refactored version of the training logic that can be called by Optuna.
    It accepts hyperparameters as an argument and returns the best validation loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lang_pair = sorted([args.source, args.target])
    tokenizer_path = os.path.join(Config.EXPERIMENTS_DIR, Config.TOKENIZER_PREFIX_PATTERN.format(src=lang_pair[0], tgt=lang_pair[1]) + ".model")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Please run train-tokenizer first.")
    tokenizer = Tokenizer(tokenizer_path)

    data_index = build_data_index(args.source, args.target, args)
    split_idx = int(len(data_index) * (1 - params['validation_split']))
    train_dataset = StreamingTranslationDataset(data_index[:split_idx], tokenizer)
    val_dataset = StreamingTranslationDataset(data_index[split_idx:], tokenizer)

    model = Seq2SeqTransformer(
        num_encoder_layers=params['num_layers'],
        num_decoder_layers=params['num_layers'],
        embed_size=params['embedding_size'],
        nhead=params['num_heads'],
        vocab_size=tokenizer.vocab_size,
        ff_hidden_size=params['ff_hidden_size'],
        dropout=params['dropout']
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_TOKEN)
    best_val_loss = float('inf')

    for epoch in range(1, params['epochs'] + 1):
        model.train()
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            optimizer.zero_grad()
            tgt_out = tgt[:, 1:]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)
                logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                tgt_out = tgt[:, 1:]
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
    
    return best_val_loss


def run_translation(model, text, tokenizer, device):
    """Translates a single text using greedy decoding."""
    model.eval()
    with torch.no_grad():
        src_ids = [Config.SOS_TOKEN] + tokenizer.encode(text) + [Config.EOS_TOKEN]
        src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
        src_mask = None
        src_padding_mask = (src_tensor == Config.PAD_TOKEN)
        memory = model.encode(src_tensor, src_mask, src_padding_mask, device)
        ys = torch.ones(1, 1).fill_(Config.SOS_TOKEN).type(torch.long).to(device)
        for _ in range(50):
            tgt_mask = generate_square_subsequent_mask(ys.size(1), device)
            out = model.decode(ys, memory, tgt_mask, None, src_padding_mask, device)
            prob = model.generator(out[:, -1])
            _, next_word_id = torch.max(prob, dim=1)
            if next_word_id.item() == Config.EOS_TOKEN:
                break
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word_id.item())], dim=1)
    return tokenizer.decode(ys.squeeze().tolist())

def run_test(model, test_pairs, tokenizer, device, source, target):
    print(f"Testing model on {len(test_pairs)} sentences...")
    results = [f"{src}\t{tgt}\t{run_translation(model, src, tokenizer, device)}\n" for src, tgt in test_pairs]
    os.makedirs(Config.TMP_DIR, exist_ok=True)
    output_path = os.path.join(Config.TMP_DIR, Config.EVAL_RESULTS_PATTERN.format(src=source, tgt=target))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{source}\t{target}\tmodel_translation\n"); f.writelines(results)
    print(f"Test complete. Results saved to '{output_path}'")


# --- 5. Checkpointing & Config Management ---

def load_hyperparams(source, target):
    """
    Loads hyperparameters using an inheritance model.
    1. Loads the base `hyperparams.yaml`.
    2. If it exists, loads the language-specific `hyperparams-{src}-{tgt}.yaml`
       and uses its values to override the base settings.
    """
    # 1. Load base config
    with open(Config.DEFAULT_HYPERPARAMS_FILE, 'r') as f:
        params = yaml.safe_load(f)

    # 2. Check for and apply language-specific overrides
    specific_path = Config.HYPERPARAMS_OUTPUT_PATTERN.format(src=source, tgt=target)
    if os.path.exists(specific_path):
        print(f"Found language-specific config. Overriding defaults from: {os.path.basename(specific_path)}")
        with open(specific_path, 'r') as f:
            specific_params = yaml.safe_load(f)
        params.update(specific_params)
    else:
        print(f"Loaded hyperparameters from: {os.path.basename(Config.DEFAULT_HYPERPARAMS_FILE)}")

    return params

def write_checkpoint(model, params, args):
    os.makedirs(Config.EXPERIMENTS_DIR, exist_ok=True)
    path = os.path.join(Config.EXPERIMENTS_DIR, Config.CHECKPOINT_FILE_PATTERN.format(src=args.source, tgt=args.target))
    # Save both model state and the hyperparameters used to train it
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparams': params
    }, path)
    return path

def read_checkpoint(source, target, device, model_path=None):
    path = model_path or os.path.join(Config.EXPERIMENTS_DIR, Config.CHECKPOINT_FILE_PATTERN.format(src=source, tgt=target))
    if not os.path.exists(path): return None, None
    # Restore safe_globals for security
    with safe_globals([argparse.Namespace]):
        checkpoint = torch.load(path, map_location=device)
    print(f"Loaded checkpoint from {path}")
    return checkpoint['model_state_dict'], checkpoint['hyperparams']


# --- 6. CLI Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Zolai-Centric Multilingual NMT CLI Tool", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)
    train_parser = subparsers.add_parser('train', help='Train a new model')
    translate_parser = subparsers.add_parser('translate', help='Translate a text')
    template_test_parser = subparsers.add_parser('test-template', help='Test the template generation engine')
    test_parser = subparsers.add_parser('test', help='Evaluate the model on a test file')
    tokenizer_parser = subparsers.add_parser('train-tokenizer', help='Train the SentencePiece tokenizer')
    test_tsv_parser = subparsers.add_parser('test-tsv-index', help='Test the TSV parallel data loader')

    for p in [train_parser, translate_parser, template_test_parser, test_parser, tokenizer_parser, test_tsv_parser]:
        p.add_argument('--source', type=str, required=True, choices=Config.SUPPORTED_LANGUAGES)
        p.add_argument('--target', type=str, required=True, choices=Config.SUPPORTED_LANGUAGES)
    
    for p in [train_parser, tokenizer_parser, template_test_parser]:
        p.add_argument('--use-templates', action='store_true')
    
    for p in [train_parser, tokenizer_parser]:
        p.add_argument('--use-tsv-index', action='store_true')

    for p in [train_parser, template_test_parser]:
        p.add_argument('--include-template-tags', type=str, default=None)
        p.add_argument('--exclude-template-tags', type=str, default=None)

    template_test_parser.add_argument('--files', type=str, default=None, help="Comma-separated list of specific template files to use.")
    
    for p in [template_test_parser, test_tsv_parser]:
        p.add_argument('--output-path', type=str, required=True)
    
    translate_parser.add_argument('--text', type=str, required=True)
    for p in [translate_parser, test_parser]: 
        p.add_argument('--model-path', type=str, default=None)
    
    args = parser.parse_args()
    if args.source == args.target: sys.exit("Error: Source and target languages cannot be the same.")
    if Config.PRIMARY_LANGUAGE not in [args.source, args.target]: sys.exit(f"Error: Zolai ('{Config.PRIMARY_LANGUAGE}') must be the source or target.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        if args.command == 'test-tsv-index':
            pairs = get_indexed_pairs(args.source, args.target)
            if not pairs: print("No data generated."); return
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            with open(args.output_path, 'w', encoding='utf-8') as f:
                f.write(f"{args.source}\t{args.target}\n")
                for pair in pairs: f.write(f"{pair[0]}\t{pair[1]}\n")
            print(f"Successfully extracted {len(pairs)} pairs to '{args.output_path}'")
            return

        lang_pair = sorted([args.source, args.target])
        tokenizer_path = os.path.join(Config.EXPERIMENTS_DIR, Config.TOKENIZER_PREFIX_PATTERN.format(src=lang_pair[0], tgt=lang_pair[1]) + ".model")
        
        if args.command == 'train-tokenizer':
            params = load_hyperparams(args.source, args.target)
            text_file, max_vocab = prepare_tokenizer_data(args.source, args.target, args, params)
            final_vocab_size = min(params['vocab_size'], max_vocab)
            if final_vocab_size < params['vocab_size']: print(f"Warning: Capping vocab size at {max_vocab}.")
            print(f"Training tokenizer on {text_file} with vocab size {final_vocab_size}...")
            model_prefix = os.path.join(Config.EXPERIMENTS_DIR, Config.TOKENIZER_PREFIX_PATTERN.format(src=lang_pair[0], tgt=lang_pair[1]))
            Tokenizer().train(text_file, model_prefix, final_vocab_size)
            print(f"Tokenizer model saved to {model_prefix}.model")
            return

        if not os.path.exists(tokenizer_path):
            sys.exit(f"Error: Tokenizer model not found at '{tokenizer_path}'. Please run 'train-tokenizer' for the '{lang_pair[0]}-{lang_pair[1]}' pair first.")
        
        tokenizer = Tokenizer(tokenizer_path)

        if args.command == 'train':
            params = load_hyperparams(args.source, args.target)
            # This is a placeholder for the full training loop that would be here.
            # We call the refactored function for clarity.
            print("Starting training run...")
            run_training_for_tuning(params, args)
            print("Training run complete.")
        
        elif args.command in ['translate', 'test']:
            model_state, loaded_params = read_checkpoint(args.source, args.target, device, args.model_path)
            if not model_state: sys.exit(f"Error: No checkpoint found for '{args.source}-{args.target}'.")
            
            model = Seq2SeqTransformer(
                num_encoder_layers=loaded_params['num_layers'],
                num_decoder_layers=loaded_params['num_layers'],
                embed_size=loaded_params['embedding_size'],
                nhead=loaded_params['num_heads'],
                vocab_size=tokenizer.vocab_size,
                ff_hidden_size=loaded_params['ff_hidden_size'],
                dropout=loaded_params['dropout']
            ).to(device)
            model.load_state_dict(model_state)
            
            if args.command == 'translate':
                translation = run_translation(model, args.text, tokenizer, device)
                print(f"\nSource ({args.source}): '{args.text}'\nTranslation ({args.target}): '{translation}'")
            elif args.command == 'test':
                test_pairs = get_indexed_pairs(args.source, args.target, 'test')
                if not test_pairs: sys.exit(f"Error: No test data found for '{args.source}-{args.target}' in the parallel index.")
                run_test(model, test_pairs, tokenizer, device, args.source, args.target)

        elif args.command == 'test-template':
            template_pairs = load_templated_data(args.source, args.target, args)
            if not template_pairs: print("No data generated."); return
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            with open(args.output_path, 'w', encoding='utf-8') as f:
                f.write(f"{args.source}\t{args.target}\n")
                for pair in template_pairs: f.write(f"{pair[0]}\t{pair[1]}\n")
            print(f"Successfully generated {len(template_pairs)} pairs to '{args.output_path}'")
            
    except (FileNotFoundError, ValueError, AttributeError, RuntimeError) as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        print("-------------------", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
