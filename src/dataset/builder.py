# -----------------------------------------------------------------------------
# File: src/dataset/builder.py
#
# Description:
#   This script processes data for a specific translation direction. It can now
#   be pointed to a specific configuration directory for testing purposes.
# -----------------------------------------------------------------------------

import yaml
import random
import sys
import argparse
import re
from pathlib import Path
from tokenizers import Tokenizer

def load_config(config_dir: Path):
    """
    Loads all .yaml files from a given config directory, merges them,
    and robustly resolves all nested ${group.key} placeholders.
    """
    config = {}
    for config_file in sorted(config_dir.glob('*.yaml')):
        with open(config_file, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            if content:
                for key, value in content.items():
                    if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                        config[key].update(value)
                    else:
                        config[key] = value

    config_str = yaml.dump(config)
    for _ in range(5):
        placeholders = set(re.findall(r'\$\{(.*?)\}', config_str))
        if not placeholders: break
        for p_str in placeholders:
            if p_str == 'paths.root':
                root_path_val = str(Path(config['paths']['root']).resolve())
                config_str = config_str.replace(f'${{{p_str}}}', root_path_val)
                continue
            try:
                lookup_config = yaml.safe_load(config_str)
                group, key = p_str.split('.')
                value = lookup_config.get(group, {}).get(key)
                if isinstance(value, str) and not re.search(r'\$\{(.*?)\}', value):
                    config_str = config_str.replace(f'${{{p_str}}}', value)
            except (ValueError, KeyError): continue
    
    return yaml.safe_load(config_str)

def main():
    """Main function to process the dataset."""
    parser = argparse.ArgumentParser(description="Dataset Processing Module")
    parser.add_argument('--config-dir', type=str, default='./config', help="Path to the configuration directory.")
    parser.add_argument('--src_lang', type=str, required=True, help="Source language code")
    parser.add_argument('--tgt_lang', type=str, required=True, help="Target language code")
    args = parser.parse_args()

    print(f"--- Processing Dataset: {args.src_lang} -> {args.tgt_lang} ---")
    
    cfg = load_config(Path(args.config_dir))
    data_cfg = cfg['data_paths']
    proc_cfg = cfg['processing']
    
    processed_path = Path(data_cfg['processed'])
    processed_path.mkdir(parents=True, exist_ok=True)

    tokenizer_path = Path(data_cfg['tokenizers'])
    src_tokenizer = Tokenizer.from_file(str(tokenizer_path / cfg['tokenizer']['tokenizer_file'].format(lang=args.src_lang)))
    tgt_tokenizer = Tokenizer.from_file(str(tokenizer_path / cfg['tokenizer']['tokenizer_file'].format(lang=args.tgt_lang)))

    parallel_path = Path(data_cfg['parallel_base'])
    with open(parallel_path / 'index.yaml', 'r', encoding='utf-8') as f:
        index = yaml.safe_load(f)
    
    src_lines, tgt_lines = [], []
    for basename in index.get('corpora', []):
        src_file = parallel_path / f"{basename}.{args.src_lang}"
        tgt_file = parallel_path / f"{basename}.{args.tgt_lang}"
        if src_file.exists() and tgt_file.exists():
            with open(src_file, 'r', encoding='utf-8') as fs:
                src_lines.extend(fs.readlines())
            with open(tgt_file, 'r', encoding='utf-8') as ft:
                tgt_lines.extend(ft.readlines())
    
    combined = list(zip(src_lines, tgt_lines))
    random.seed(proc_cfg['shuffle_seed'])
    random.shuffle(combined)
    
    total_size = len(combined)
    val_size = int(total_size * proc_cfg['split_ratio']['val']) or (1 if total_size > 1 else 0)
    test_size = int(total_size * proc_cfg['split_ratio']['test']) or (1 if total_size > 1 else 0)
    train_size = total_size - val_size - test_size

    splits = {
        'train': combined[:train_size],
        'val': combined[train_size:train_size + val_size],
        'test': combined[train_size + val_size:]
    }

    for split_name, split_data in splits.items():
        if not split_data: continue
        print(f"  - Writing {len(split_data)} sentences to {split_name} split.")
        with open(processed_path / f"{split_name}.{args.src_lang}", 'w', encoding='utf-8') as fs, \
             open(processed_path / f"{split_name}.{args.tgt_lang}", 'w', encoding='utf-8') as ft:
            for src_line, tgt_line in split_data:
                if src_line.strip():
                    src_encoded = src_tokenizer.encode(src_line.strip())
                    fs.write(' '.join(map(str, src_encoded.ids)) + '\n')
                if tgt_line.strip():
                    tgt_encoded = tgt_tokenizer.encode(tgt_line.strip())
                    ft.write(' '.join(map(str, tgt_encoded.ids)) + '\n')
    
    print("[OK] Dataset processing complete.")

if __name__ == '__main__':
    main()
