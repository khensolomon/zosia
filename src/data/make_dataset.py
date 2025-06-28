# Handle data cleaning, tokenization, and saving processed data.

import os
import argparse
import yaml # Make sure PyYAML is installed: pip install PyYAML
import pandas as pd
import torch
import sentencepiece as spm
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.utils.general_utils import load_config

def clean_text(text: str, config: dict) -> str:
    """Applies basic text cleaning based on config."""
    if pd.isna(text): # Handle potential NaN values
        return ""
    text = str(text) # Ensure text is string
    if config.get("lowercase", False):
        text = text.lower()
    if config.get("normalize_punctuation", True):
        # Basic normalization, extend as needed
        text = text.replace("'", " ' ").replace('"', ' " ').replace('.', ' . ').replace(',', ' , ').strip()
    if config.get("strip_whitespace", True):
        text = " ".join(text.split())
    return text

def train_tokenizer(data_paths: list, vocab_dir: str, prefix: str, vocab_size: int):
    """Trains a SentencePiece tokenizer on combined data."""
    combined_corpus_path = os.path.join(vocab_dir, "combined_corpus.txt")
    os.makedirs(vocab_dir, exist_ok=True)
    logger.info(f"Preparing combined corpus for tokenizer training at: {combined_corpus_path}")

    # Check if data_paths is empty
    if not data_paths:
        logger.warning("No data paths provided for tokenizer training. Skipping tokenizer training.")
        return False # Indicate that tokenizer training was skipped

    with open(combined_corpus_path, 'w', encoding='utf-8') as outfile:
        for path in data_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        cleaned_line = " ".join(line.strip().split())
                        if cleaned_line:
                            outfile.write(cleaned_line + '\n')
            else:
                logger.warning(f"Data file not found for tokenizer training: {path}. Skipping.")

    if not os.path.getsize(combined_corpus_path) > 0:
        logger.error(f"Combined corpus file is empty: {combined_corpus_path}. Cannot train tokenizer.")
        return False

    cmd = (
        f"--input={combined_corpus_path} --model_prefix={os.path.join(vocab_dir, prefix)} "
        f"--vocab_size={vocab_size} --character_coverage=1.0 --model_type=bpe "
        f"--unk_id=0 --bos_id=1 --eos_id=2 --pad_id=3 "
        f"--unk_piece=<unk> --bos_piece=<s> --eos_piece=</s> --pad_piece=<pad>"
    )
    logger.info("Training SentencePiece tokenizer...")
    logger.info(f"SentencePiece training command: {cmd}")
    try:
        spm.SentencePieceTrainer.train(cmd)
        logger.info(f"SentencePiece tokenizer trained and saved to {vocab_dir}/{prefix}.model")
        return True # Indicate success
    except Exception as e:
        logger.error(f"Error during SentencePiece tokenizer training: {e}")
        raise
    finally:
        if os.path.exists(combined_corpus_path):
            os.remove(combined_corpus_path)
            logger.info(f"Cleaned up temporary combined corpus file: {combined_corpus_path}")

def tokenize_and_save(file_path: str, sp_model: spm.SentencePieceProcessor, output_path: str, max_seq_len: int):
    """Tokenizes a single file and saves token IDs as a PyTorch tensor."""
    if not os.path.exists(file_path):
        logger.warning(f"Input file not found: {file_path}. Skipping tokenization.")
        return None

    token_ids_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f"Tokenizing {os.path.basename(file_path)}"):
            ids = sp_model.encode_as_ids(line.strip())
            ids = ids[:max_seq_len - 2] # Leave space for SOS and EOS (added later in Dataset)
            token_ids_list.append(ids)

    # Save as a list of lists of token IDs
    torch.save(token_ids_list, output_path)
    logger.info(f"Tokenized data saved to {output_path}")
    return output_path

# --- NEW HELPER FUNCTIONS ---
def load_yaml_catalog(catalog_full_path):
    """Loads a YAML catalog file."""
    if not os.path.exists(catalog_full_path):
        logger.error(f"Catalog file not found: {catalog_full_path}")
        raise FileNotFoundError(f"Catalog file not found: {catalog_full_path}")
    with open(catalog_full_path, 'r') as file:
        catalog = yaml.safe_load(file)
    return catalog

def get_paths_from_monolingual_index(index_full_path, data_root_dir):
    """Reads a CSV index file for monolingual data and returns full paths."""
    paths = []
    if not os.path.exists(index_full_path):
        logger.warning(f"Monolingual index file not found: {index_full_path}. No monolingual data will be added for tokenization.")
        return paths
    
    try:
        df = pd.read_csv(index_full_path)
        if 'filename' in df.columns:
            for filename in df['filename']:
                # The index.csv path is data/monolingual/zo/index.csv
                # We need the full path to the actual data files.
                # Assuming 'filename' column contains paths relative to data/monolingual/zo/
                # e.g., 'bible.txt' means data/monolingual/zo/bible.txt
                # So the data_root_dir for these should be the directory of the index file.
                paths.append(os.path.join(os.path.dirname(index_full_path), filename))
        else:
            logger.warning(f"Column 'filename' not found in {index_full_path}. Check CSV format.")
    except Exception as e:
        logger.error(f"Error reading monolingual index file {index_full_path}: {e}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Prepare data for NMT training.")
    parser.add_argument("--config", type=str, default="config/data_config.yaml",
                        help="Path to the data configuration YAML file. Default: config/data_config.yaml")
    args = parser.parse_args()

    try:
        data_config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        logger.error(f"Please ensure '{args.config}' exists or provide a correct path.")
        return
    except Exception as e:
        logger.error(f"Error loading configuration from {args.config}: {e}")
        return

    # --- Retrieve common paths from config ---
    raw_data_dir = data_config.get("raw_data_dir")
    processed_data_dir = data_config.get("processed_data_dir")
    vocab_dir = data_config.get("vocab_dir")
    tokenizer_prefix = data_config.get("tokenizer_prefix")
    max_sequence_length = data_config.get("max_sequence_length", 128)
    vocab_size = data_config.get("vocab_size")

    # --- Validate essential configs ---
    if vocab_size is None:
        logger.error("Error: 'vocab_size' not found in data_config.yaml. Please specify it.")
        return
    if raw_data_dir is None:
        logger.error("Error: 'raw_data_dir' not found in data_config.yaml.")
        return

    # Ensure output directories exist
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(vocab_dir, exist_ok=True)
    logger.info(f"Ensured processed data directory exists: {processed_data_dir}")
    logger.info(f"Ensured vocabulary directory exists: {vocab_dir}")

    # --- Load Raw Data Catalog (YAML) ---
    raw_data_catalog_file = data_config.get("raw_data_catalog_file")
    if not raw_data_catalog_file:
        logger.error("Error: 'raw_data_catalog_file' not found in data_config.yaml. This is now required.")
        return
    
    raw_catalog_full_path = os.path.join(raw_data_dir, raw_data_catalog_file)
    try:
        raw_data_catalog = load_yaml_catalog(raw_catalog_full_path)
        logger.info(f"Loaded raw data catalog from: {raw_catalog_full_path}")
    except FileNotFoundError:
        logger.error(f"Raw data catalog file not found at {raw_catalog_full_path}.")
        return
    except Exception as e:
        logger.error(f"Error loading raw data catalog: {e}")
        return

    # --- Collect all text files for tokenizer training ---
    all_text_files_for_spm = []

    # Add parallel data from the YAML catalog
    if 'parallel_data' in raw_data_catalog:
        for split_type in ['train', 'val', 'test']: # Iterate all relevant splits for tokenizer training
            if split_type in raw_data_catalog['parallel_data']:
                for item in raw_data_catalog['parallel_data'][split_type]:
                    full_path_en = os.path.join(raw_data_dir, item['path_en'])
                    full_path_zo = os.path.join(raw_data_dir, item['path_zo'])
                    all_text_files_for_spm.extend([full_path_en, full_path_zo])
    
    # Add monolingual data from index.csv (new path)
    monolingual_data_dir = data_config.get("monolingual_data_dir")
    monolingual_zo_index_file = data_config.get("monolingual_zo_index_file")
    
    if monolingual_data_dir and monolingual_zo_index_file:
        monolingual_zo_index_full_path = os.path.join(monolingual_data_dir, monolingual_zo_index_file)
        # Assuming files in index.csv are relative to data/monolingual/zo/
        monolingual_zo_files_from_index = get_paths_from_monolingual_index(
            monolingual_zo_index_full_path, 
            os.path.dirname(monolingual_zo_index_full_path) # Pass the directory of the index as root for its files
        )
        all_text_files_for_spm.extend(monolingual_zo_files_from_index)
        logger.info(f"Added {len(monolingual_zo_files_from_index)} files from monolingual index for SPM training.")

    # Add any direct monolingual files from data_config (like monolingual_en_file)
    monolingual_en_file = data_config.get("monolingual_en_file")
    if monolingual_en_file:
        full_monolingual_en_path = os.path.join(raw_data_dir, monolingual_en_file)
        all_text_files_for_spm.append(full_monolingual_en_path)
        logger.info(f"Added {full_monolingual_en_path} for SPM training.")


    # 2. Train SentencePiece tokenizer
    tokenizer_trained = train_tokenizer(all_text_files_for_spm, vocab_dir, tokenizer_prefix, vocab_size)
    if not tokenizer_trained:
        logger.error("SentencePiece tokenizer training failed or was skipped. Cannot proceed.")
        return

    # Load trained tokenizer
    tokenizer_model_path = os.path.join(vocab_dir, f"{tokenizer_prefix}.model")
    try:
        sp_model = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
        logger.info(f"Tokenizer loaded successfully from {tokenizer_model_path}")
        logger.info(f"Tokenizer vocabulary size: {sp_model.get_piece_size()}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer model from {tokenizer_model_path}: {e}")
        return

    # 3. Process and tokenize parallel data using the catalog
    splits = ['train', 'val', 'test']
    for split in splits:
        if 'parallel_data' in raw_data_catalog and split in raw_data_catalog['parallel_data']:
            split_items = raw_data_catalog['parallel_data'][split]
            if not split_items:
                logger.info(f"No files specified for {split} split in the catalog. Skipping.")
                continue

            # Iterate through each item (potentially multiple file pairs) for the current split
            for item_idx, item in enumerate(split_items):
                src_file_relative = item['path_en']
                tgt_file_relative = item['path_zo']

                src_path_full = os.path.join(raw_data_dir, src_file_relative)
                tgt_path_full = os.path.join(raw_data_dir, tgt_file_relative)

                logger.info(f"Processing {split} data source '{item.get('description', f'item {item_idx}')}' from '{src_file_relative}' and '{tgt_file_relative}'...")

                # Output filenames will reflect the original split name (e.g., train_token_ids.zo.pt)
                # If you need separate processed files for each sub-item in the catalog,
                # you'd modify the output path to include item_idx or item['description']
                output_src_path = os.path.join(processed_data_dir, f"{split}_token_ids.zo.pt")
                output_tgt_path = os.path.join(processed_data_dir, f"{split}_token_ids.en.pt")

                # Note: If multiple files for the same split (e.g., train) are listed in the YAML,
                # this current logic will overwrite the .pt file unless you modify the output_path.
                # For robust handling of multiple files *per split* in the catalog, you might want to:
                # 1. Tokenize each source separately, then concatenate the tokenized lists.
                # 2. Use a unique output filename for each catalog entry if you want to keep them separate.
                # For now, we'll assume the primary use is one set of train/val/test per catalog,
                # or that `tokenize_and_save` handles appending/concatenating internally.
                # The provided `tokenize_and_save` *overwrites* so we need to be careful if
                # multiple sources for 'train' are listed.
                # For simplicity, I'll adjust the logic to aggregate lines from multiple sources
                # for a single split before saving one .pt file.

                all_src_lines_for_split = []
                all_tgt_lines_for_split = []

                if os.path.exists(src_path_full) and os.path.exists(tgt_path_full):
                    with open(src_path_full, 'r', encoding='utf-8') as f_src:
                        src_content = f_src.readlines()
                    with open(tgt_path_full, 'r', encoding='utf-8') as f_trg:
                        tgt_content = f_trg.readlines()

                    if len(src_content) != len(tgt_content):
                        logger.warning(f"Mismatch in line count for {split} data item (from {src_file_relative}). Src: {len(src_content)}, Tgt: {len(tgt_content)}. Truncating to min length.")
                        min_len = min(len(src_content), len(tgt_content))
                        src_content = src_content[:min_len]
                        tgt_content = tgt_content[:min_len]

                    all_src_lines_for_split.extend(src_content)
                    all_tgt_lines_for_split.extend(tgt_content)
                else:
                    logger.warning(f"Skipping catalog item for {split}: One or both files not found: {src_path_full}, {tgt_path_full}")
            
            # --- Now tokenize and save the aggregated data for the split ---
            if all_src_lines_for_split and all_tgt_lines_for_split:
                tokenized_src_ids = []
                tokenized_trg_ids = []

                logger.info(f"Tokenizing aggregated {split} data ({len(all_src_lines_for_split)} pairs)...")
                for src_line, trg_line in tqdm(zip(all_src_lines_for_split, all_tgt_lines_for_split), total=len(all_src_lines_for_split), desc=f"Tokenizing {split} aggregated"):
                    src_ids = sp_model.encode_as_ids(src_line.strip())
                    trg_ids = sp_model.encode_as_ids(trg_line.strip())
                    tokenized_src_ids.append(src_ids[:max_sequence_length - 2])
                    tokenized_trg_ids.append(trg_ids[:max_sequence_length - 2])
                
                torch.save(tokenized_src_ids, os.path.join(processed_data_dir, f"{split}_token_ids.zo.pt"))
                torch.save(tokenized_trg_ids, os.path.join(processed_data_dir, f"{split}_token_ids.en.pt"))
                logger.info(f"Aggregated tokenized {split} data saved to {processed_data_dir}/{split}_token_ids.zo.pt and .en.pt")
            else:
                logger.info(f"No valid data found to process for {split} split from catalog.")
        else:
            logger.info(f"Skipping {split} data as no 'parallel_data' entry for it in the catalog.")

    logger.info("Data preparation complete.")

if __name__ == "__main__":
    main()