"""
Zolai-NMT Data Preprocessing Module
version: 2025.08.08.1535

This module contains all functions related to loading, generating, and preparing
data for the NMT model. It handles TSV file indexing, template-based data
generation, and preparation of data for tokenizer training.
"""
import os
import io
import random
import re
import itertools
import yaml
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# --- 1. Data Loading & Generation ---

def get_indexed_pairs(cfg, source_lang, target_lang, data_type='train'):
    """Loads parallel data from a list of TSV files defined in datasets.yaml."""
    index_file = cfg.data.paths.datasets_index_file
    if not os.path.exists(index_file):
        print(f"Warning: Datasets index file not found at {index_file}")
        return []

    with open(index_file, 'r', encoding='utf-8') as f:
        index = yaml.safe_load(f)

    direction_key = f"{source_lang}-{target_lang}"
    direction_config = index.get(direction_key, {})

    # Determine the correct directory for train/test files
    dir_key = f"{data_type}_dir"
    data_dir = index.get(dir_key, os.path.join(cfg.data.paths.corpus_dir, data_type))

    # Combine specific and shared basenames, ensuring uniqueness
    specific_basenames = direction_config.get(data_type, [])
    shared_basenames = index.get("shared", {}).get(data_type, [])
    basenames = list(dict.fromkeys(specific_basenames + shared_basenames))

    all_pairs = []
    print(f"Loading TSV parallel data for '{direction_key}' ({data_type} set)...")
    for item in basenames:
        file_path = os.path.join(data_dir, f"{item}.tsv")

        if not os.path.exists(file_path):
            print(f"  - Warning: File '{file_path}' not found. Skipping.")
            continue

        with io.open(file_path, encoding='utf-8') as f:
            try:
                header = f.readline().strip().split('\t')
                src_idx = header.index(source_lang)
                tgt_idx = header.index(target_lang)
                print(f"  - Loading from '{os.path.basename(file_path)}'...")
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) > max(src_idx, tgt_idx):
                        all_pairs.append((parts[src_idx], parts[tgt_idx]))
            except (ValueError, IndexError):
                print(f"  - Skipping '{os.path.basename(file_path)}' (missing required language columns or malformed line).")
                continue
    return all_pairs

def load_templated_data(cfg, source_lang, target_lang, args):
    """Generates sentence pairs by crawling and processing YAML template files."""
    template_dir = cfg.data.paths.template_dir
    if not os.path.exists(template_dir): return []

    # Safely get tag arguments, defaulting to None if not present
    include_tags_str = getattr(args, 'include_tags', None)
    exclude_tags_str = getattr(args, 'exclude_tags', None)
    
    include_tags = set(include_tags_str.split(',')) if include_tags_str else None
    exclude_tags = set(exclude_tags_str.split(',')) if exclude_tags_str else None

    all_pairs = []
    print(f"Loading templated data for '{source_lang}-{target_lang}'...")
    # Sort filenames to ensure deterministic order
    for filename in sorted(os.listdir(template_dir)):
        if not filename.endswith((".yaml", ".yml")): continue
        
        file_path = os.path.join(template_dir, filename)
        
        # Tag filtering logic
        if include_tags or exclude_tags:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = yaml.safe_load(f)
                    file_tags = set(data.get('tags', []))
                    if include_tags and not file_tags.intersection(include_tags): 
                        continue
                    if exclude_tags and file_tags.intersection(exclude_tags):
                        continue
                except yaml.YAMLError: 
                    continue
        
        print(f"  - Generating from '{filename}'...")
        all_pairs.extend(generate_from_template_file(cfg, file_path, source_lang, target_lang))
    return all_pairs

def capitalize_sentence(sentence):
    """Capitalizes the first letter of a sentence, ignoring 'i'."""
    if not sentence: return ""
    parts = sentence.split(' ', 1)
    first_word = parts[0]
    if first_word.lower() != 'i':
        first_word = first_word.capitalize()
    return ' '.join([first_word] + parts[1:]) if len(parts) > 1 else first_word

def generate_from_template_file(cfg, file_path, source_lang, target_lang):
    """The core template generation logic for a single YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f: template_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Warning: Could not parse YAML file {file_path}. Error: {e}"); return []
    
    # Handle imports from the shared template directory
    if 'import' in template_data:
        merged_data = {}
        for import_file in template_data['import']:
            import_path = os.path.join(cfg.data.paths.shared_template_dir, import_file)
            if os.path.exists(import_path):
                with open(import_path, 'r', encoding='utf-8') as f:
                    shared_data = yaml.safe_load(f)
                    for key, value in shared_data.items():
                        if key not in merged_data:
                            merged_data[key] = value
                        elif isinstance(merged_data.get(key), list) and isinstance(value, list):
                            merged_data[key].extend(value)
            else:
                print(f"Warning: Imported file not found: {import_path}")
        # Merge local data over imported data
        for key, value in template_data.items():
            if key in merged_data and isinstance(merged_data.get(key), list) and isinstance(value, list):
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
            if isinstance(values, dict) and source_lang in values and target_lang in values:
                src_vals, tgt_vals = values[source_lang], values[target_lang]
                canonical_keys = values.get('en', src_vals)
                if not (len(src_vals) == len(tgt_vals) == len(canonical_keys)):
                    valid_template = False; break
                for i in range(len(src_vals)):
                    metadata_key = canonical_keys[i]
                    tags_raw = template_data.get('metadata', {}).get(p, {}).get(metadata_key, {}).get('tags', [])
                    tags = [f"{k}:{v}" if isinstance(t, dict) else str(t) for t in tags_raw for k, v in (t.items() if isinstance(t, dict) else [(None, t)]) if v is not None]
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
                                    if isinstance(condition, str) and condition in dependent_tags: matched = True
                                    elif isinstance(condition, dict):
                                        if 'all_of' in condition and set(condition['all_of']).issubset(dependent_tags): matched = True
                                        elif 'any_of' in condition and any(t in dependent_tags for t in condition['any_of']): matched = True
                                        elif 'none_of' in condition and not any(t in dependent_tags for t in condition['none_of']): matched = True
                                    
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

# --- 2. Data Preparation & Datasets ---

def prepare_tokenizer_data(cfg, source_lang, target_lang, args):
    """Aggregates all specified data sources into a single text file for tokenizer training."""
    all_sentences = []
    if args.use_parallel:
        for src, tgt in [(source_lang, target_lang), (target_lang, source_lang)]:
            train_pairs = get_indexed_pairs(cfg, src, tgt, 'train')
            test_pairs = get_indexed_pairs(cfg, src, tgt, 'test')
            all_sentences.extend([p[0] for p in train_pairs] + [p[1] for p in train_pairs])
            all_sentences.extend([p[0] for p in test_pairs] + [p[1] for p in test_pairs])
    if args.use_template:
        for src, tgt in [(source_lang, target_lang), (target_lang, source_lang)]:
            template_pairs = load_templated_data(cfg, src, tgt, args)
            all_sentences.extend([p[0] for p in template_pairs] + [p[1] for p in template_pairs])

    # Sort the collected sentences to ensure a deterministic order for hashing
    all_sentences.sort()

    os.makedirs(cfg.data.paths.tmp_dir, exist_ok=True)
    lang_pair_str = f"{source_lang}-{target_lang}"
    # Use the nested config path: cfg.app.filenames
    output_path = os.path.join(cfg.data.paths.tmp_dir, cfg.app.filenames.tokenizer_input.format(lang_pair=lang_pair_str))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_sentences))
    
    unique_words = len(set(" ".join(all_sentences).split()))
    return output_path, unique_words

def build_data_index(cfg, source_lang, target_lang, args):
    """Builds a combined index of sentence pairs from all specified sources."""
    index = []
    if args.use_parallel:
        index.extend(get_indexed_pairs(cfg, source_lang, target_lang, 'train'))
    if args.use_template:
        index.extend(load_templated_data(cfg, source_lang, target_lang, args))
    
    if not index:
        raise ValueError(f"No data found for the pair '{source_lang}-{target_lang}' from any source.")
    
    random.shuffle(index)
    return index

class StreamingTranslationDataset(Dataset):
    """A PyTorch Dataset that tokenizes sentence pairs on the fly."""
    def __init__(self, data_index, tokenizer, cfg):
        self.data_index = data_index
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        src, tgt = self.data_index[idx]
        # Use the nested config path: cfg.app.tokens
        src_ids = [self.cfg.app.tokens.sos] + self.tokenizer.encode(src) + [self.cfg.app.tokens.eos]
        tgt_ids = [self.cfg.app.tokens.sos] + self.tokenizer.encode(tgt) + [self.cfg.app.tokens.eos]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch, pad_token_id):
    """Pads sequences in a batch to the same length."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_token_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_token_id)
    return src_padded, tgt_padded
