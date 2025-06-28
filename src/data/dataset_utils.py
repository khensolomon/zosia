# Contains PyTorch Dataset and DataLoader helpers.

import torch
from torch.utils.data import Dataset, DataLoader
import os
import yaml # NEW: Import yaml for loading catalog
import sentencepiece as spm
from typing import List, Tuple, Dict, Optional # NEW: Import Optional and Dict for type hints
import logging # NEW: Import logging

# Set up a logger for this module
logger = logging.getLogger(__name__)
# Configure logging if it hasn't been configured by the main application
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class NMTDataset(Dataset):
    """
    Custom PyTorch Dataset for Neural Machine Translation.
    Loads tokenized source and target language data and prepares it for training.
    """
    def __init__(self,
                 src_data_path: str,
                 tgt_data_path: str,
                 sp_model: spm.SentencePieceProcessor, # RESTORED: This argument is necessary
                 max_seq_len: int):
        """
        Args:
            src_data_path (str): Path to the tokenized source language data (.pt file).
            tgt_data_path (str): Path to the tokenized target language data (.pt file).
            sp_model (spm.SentencePieceProcessor): The SentencePiece model.
            max_seq_len (int): Maximum sequence length for padding/truncation.
        """
        if not os.path.exists(src_data_path):
            raise FileNotFoundError(f"Source data file not found: {src_data_path}")
        if not os.path.exists(tgt_data_path):
            raise FileNotFoundError(f"Target data file not found: {tgt_data_path}")

        logger.info(f"Loading data from {src_data_path} and {tgt_data_path}")
        self.src_data = torch.load(src_data_path)
        self.tgt_data = torch.load(tgt_data_path)
        self.max_seq_len = max_seq_len

        # Ensure consistent number of samples
        if len(self.src_data) != len(self.tgt_data):
            # Changed from raise ValueError to logging a warning and truncating
            logger.warning(
                f"Mismatch in number of samples: {len(self.src_data)} source, "
                f"{len(self.tgt_data)} target. Truncating to minimum length."
            )
            min_len = min(len(self.src_data), len(self.tgt_data))
            self.src_data = self.src_data[:min_len]
            self.tgt_data = self.tgt_data[:min_len]
        
        # The NMTDataset must receive the sp_model object to get special token IDs
        # (bos_id, eos_id, pad_id)
        self.sp_model = sp_model
        self.sos_token_id = sp_model.bos_id() # Beginning Of Sentence
        self.eos_token_id = sp_model.eos_id() # End Of Sentence
        self.pad_token_id = sp_model.pad_id() # Padding

        if self.sos_token_id == -1 or self.eos_token_id == -1 or self.pad_token_id == -1:
             raise ValueError("SentencePiece special token IDs (BOS, EOS, PAD) not found or invalid. "
                              "Ensure the tokenizer was trained with --bos_id, --eos_id, --pad_id options or handles them correctly.")
        logger.info(f"NMTDataset initialized with {len(self.src_data)} samples.")


    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_ids = self.src_data[idx]
        tgt_ids = self.tgt_data[idx] # FIXED: This was previously `self.src_data[idx]`

        # Add SOS and EOS tokens and truncate/pad
        src_tensor = self._process_sequence(src_ids)
        tgt_tensor = self._process_sequence(tgt_ids)

        return src_tensor, tgt_tensor

    def _process_sequence(self, ids: List[int]) -> torch.Tensor:
        """Adds SOS/EOS, truncates, and pads a sequence."""
        # Add SOS and EOS tokens
        processed_ids = [self.sos_token_id] + ids + [self.eos_token_id]

        # Truncate if longer than max_seq_len
        if len(processed_ids) > self.max_seq_len:
            # We want to keep BOS and EOS within the max_seq_len
            # Truncate body, then add EOS if there's space
            if self.max_seq_len > 2: # Ensure there's space for BOS and EOS at least
                processed_ids = processed_ids[:self.max_seq_len - 1] # Truncate to leave space for EOS
                processed_ids[-1] = self.eos_token_id # Ensure EOS is at the end
            else: # If max_seq_len is very small, just take the first few
                processed_ids = processed_ids[:self.max_seq_len]
        
        # Pad if shorter than max_seq_len
        padding_needed = self.max_seq_len - len(processed_ids)
        if padding_needed > 0:
            processed_ids = processed_ids + [self.pad_token_id] * padding_needed

        return torch.tensor(processed_ids, dtype=torch.long)

# NEW HELPER FUNCTION for loading YAML catalogs
def load_yaml_catalog(catalog_full_path: str) -> Dict:
    """Loads a YAML catalog file."""
    if not os.path.exists(catalog_full_path):
        logger.error(f"Catalog file not found: {catalog_full_path}")
        raise FileNotFoundError(f"Catalog file not found: {catalog_full_path}")
    with open(catalog_full_path, 'r') as file:
        catalog = yaml.safe_load(file)
    logger.debug(f"Catalog loaded: {catalog_full_path}")
    return catalog


def get_dataloaders(
    data_config: dict,
    training_config: dict,
    sp_model: spm.SentencePieceProcessor # SentencePiece model for special tokens
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Creates and returns PyTorch DataLoaders for train, validation, and test sets.
    Reads file paths from the raw data catalog specified in data_config.
    """
    processed_data_dir = data_config.get("processed_data_dir")
    raw_data_dir = data_config.get("raw_data_dir")
    raw_data_catalog_file = data_config.get("raw_data_catalog_file")
    max_seq_len = data_config.get("max_sequence_length")
    batch_size = training_config.get("batch_size")

    # Validate essential configurations
    if not all([processed_data_dir, raw_data_dir, raw_data_catalog_file, max_seq_len, batch_size is not None]):
        logger.error("Missing essential configuration parameters in data_config or training_config. "
                     "Ensure processed_data_dir, raw_data_dir, raw_data_catalog_file, max_sequence_length, and batch_size are set.")
        raise ValueError("Incomplete configuration for data loaders.")

    # Construct full path to the raw data catalog (e.g., data/raw/base_training.yaml)
    raw_catalog_full_path = os.path.join(raw_data_dir, raw_data_catalog_file)
    
    # Load the raw data catalog
    try:
        raw_data_catalog = load_yaml_catalog(raw_catalog_full_path)
        logger.info(f"Successfully loaded raw data catalog from: {raw_catalog_full_path}")
    except Exception as e:
        logger.exception(f"Failed to load raw data catalog at {raw_catalog_full_path} for dataloaders.")
        raise # Re-raise the exception as it's a critical error

    # Initialize loaders to None, as some splits might not exist
    train_loader, val_loader, test_loader = None, None, None 

    # Loop through splits to create DataLoaders
    # The language codes 'zo' and 'en' are assumed from previous context and typical NMT setups.
    # If these are dynamic, they would need to be passed in data_config.
    splits = ['train', 'val', 'test']
    for split_name in splits:
        # Check if the split exists in the catalog's 'parallel_data' section
        # The catalog structure is expected to be `parallel_data: {train: {src: ..., tgt: ...}, ...}`
        if 'parallel_data' in raw_data_catalog and split_name in raw_data_catalog['parallel_data']:
            # The make_dataset.py saves files as f"{split}_token_ids.LANG.pt"
            src_pt_filename = f"{split_name}_token_ids.zo.pt" # Assuming 'zo' for source
            tgt_pt_filename = f"{split_name}_token_ids.en.pt" # Assuming 'en' for target

            src_pt_path = os.path.join(processed_data_dir, src_pt_filename)
            tgt_pt_path = os.path.join(processed_data_dir, tgt_pt_filename)

            if os.path.exists(src_pt_path) and os.path.exists(tgt_pt_path):
                logger.info(f"Attempting to create DataLoader for '{split_name}' split using: "
                            f"'{src_pt_path}' and '{tgt_pt_path}'")
                try:
                    dataset = NMTDataset(
                        src_data_path=src_pt_path,
                        tgt_data_path=tgt_pt_path,
                        sp_model=sp_model, # Pass sp_model to NMTDataset for special token IDs
                        max_seq_len=max_seq_len
                    )
                    
                    shuffle_data = True if split_name == 'train' else False
                    
                    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=shuffle_data,
                        num_workers=0, # Set to >0 for production (e.g., os.cpu_count() or a tuned value), but 0 for easier debugging
                        pin_memory=True if torch.cuda.is_available() else False
                    )
                    
                    if split_name == 'train':
                        train_loader = dataloader
                    elif split_name == 'val':
                        val_loader = dataloader
                    elif split_name == 'test':
                        test_loader = dataloader
                    
                    logger.info(f"'{split_name}' DataLoader created with {len(dataset)} samples.")
                except Exception as e:
                    logger.error(f"Error creating DataLoader for '{split_name}' split from files "
                                 f"'{src_pt_path}' and '{tgt_pt_path}': {e}")
                    # The specific loader will remain None if an error occurs
            else:
                logger.warning(f"Processed data files for '{split_name}' not found: "
                               f"'{src_pt_path}' or '{tgt_pt_path}'. Skipping DataLoader for this split.")
        else:
            logger.info(f"No 'parallel_data' entry for '{split_name}' split in catalog. Skipping DataLoader.")

    # Provide a summary of created DataLoaders
    train_samples = len(train_loader.dataset) if train_loader else 'N/A'
    val_samples = len(val_loader.dataset) if val_loader else 'N/A'
    test_samples = len(test_loader.dataset) if test_loader else 'N/A'
    print(f"DataLoaders creation summary: Train ({train_samples} samples), "
          f"Val ({val_samples} samples), Test ({test_samples} samples)")
    
    return train_loader, val_loader, test_loader