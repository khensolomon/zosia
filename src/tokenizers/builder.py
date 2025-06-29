# -----------------------------------------------------------------------------
# File: src/tokenizers/builder.py
#
# Description:
#   This script is responsible for training the sentence tokenizers. It can now
#   be pointed to a specific configuration directory for testing purposes.
# -----------------------------------------------------------------------------

import yaml
import sys
from pathlib import Path
import argparse
import re
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

def load_config(config_dir: Path):
    """
    Loads all .yaml files from a given config directory, merges them,
    and robustly resolves all nested ${group.key} placeholders.
    """
    config = {}
    # Sort to ensure default.yaml is loaded first
    for config_file in sorted(config_dir.glob('*.yaml')):
        with open(config_file, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            if content:
                # A simple way to merge nested dictionaries
                for key, value in content.items():
                    if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                        config[key].update(value)
                    else:
                        config[key] = value

    # Iteratively replace placeholders to handle nested dependencies
    config_str = yaml.dump(config)
    for _ in range(5): # Limit iterations to prevent infinite loops
        placeholders = set(re.findall(r'\$\{(.*?)\}', config_str))
        if not placeholders:
            break
        for p_str in placeholders:
            # Special handling for paths.root to resolve it to an absolute path
            if p_str == 'paths.root':
                root_path_val = str(Path(config['paths']['root']).resolve())
                config_str = config_str.replace(f'${{{p_str}}}', root_path_val)
                continue
            
            # General placeholder replacement
            try:
                # Use a copy of the parsed config for lookups
                lookup_config = yaml.safe_load(config_str)
                group, key = p_str.split('.')
                value = lookup_config.get(group, {}).get(key)
                if isinstance(value, str):
                    config_str = config_str.replace(f'${{{p_str}}}', value)
            except (ValueError, KeyError):
                continue
    
    return yaml.safe_load(config_str)

def get_all_corpus_files(cfg, lang: str):
    """Gathers a list of all corpus files for a given language."""
    file_paths = []
    base_data_path = Path(cfg['paths']['data'])
    parallel_path = base_data_path / 'parallel_base'
    if (parallel_index_file := parallel_path / 'index.yaml').exists():
        with open(parallel_index_file, 'r', encoding='utf-8') as f:
            index = yaml.safe_load(f)
            for basename in index.get('corpora', []):
                if (file := parallel_path / f"{basename}.{lang}").exists() and file.stat().st_size > 0:
                    file_paths.append(str(file))
    if lang == 'zo':
        monolingual_path = base_data_path / 'monolingual' / 'zo'
        if (monolingual_index_file := monolingual_path / 'index.yaml').exists():
            with open(monolingual_index_file, 'r', encoding='utf-8') as f:
                index = yaml.safe_load(f)
                for filename in index.get('files', []):
                    if (file := monolingual_path / filename).exists() and file.stat().st_size > 0:
                        file_paths.append(str(file))
    return file_paths

def train_tokenizer(cfg, lang: str):
    """Trains and saves a tokenizer for a specific language."""
    print(f"--- Preparing to train tokenizer for language: '{lang}' ---")
    tokenizer_cfg = cfg['tokenizer']
    corpus_files = get_all_corpus_files(cfg, lang)
    if not corpus_files:
        print(f"\n[ERROR] No valid corpus files found for '{lang}'.")
        return False

    tokenizer = Tokenizer(WordPiece(unk_token=tokenizer_cfg['special_tokens'][0]))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=tokenizer_cfg['vocab_size'], min_frequency=tokenizer_cfg['min_frequency'], special_tokens=tokenizer_cfg['special_tokens'])
    
    print(f"Starting training for '{lang}' with {len(corpus_files)} file(s).")
    tokenizer.train(files=corpus_files, trainer=trainer)
    
    tokenizer_dir = Path(cfg['data_paths']['tokenizers'])
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    save_path = tokenizer_dir / tokenizer_cfg['tokenizer_file'].format(lang=lang)
    tokenizer.save(str(save_path))
    print(f"[OK] Tokenizer for '{lang}' saved to: {save_path}")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tokenizer Training Module")
    parser.add_argument('--config-dir', type=str, default='./config', help="Path to the configuration directory.")
    args = parser.parse_args()

    print("===================================")
    print("  ZoSia Tokenizer Training Module  ")
    print("===================================\n")

    config = load_config(Path(args.config_dir))
    success_map = {lang: train_tokenizer(config, lang) for lang in config.get('languages', [])}
    
    if all(success_map.values()):
        print("\nAll tokenizers trained successfully.")
        sys.exit(0)
    else:
        failed = [lang for lang, status in success_map.items() if not status]
        print(f"\nTokenizer training failed for: {failed}")
        sys.exit(1)
