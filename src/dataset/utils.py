# -----------------------------------------------------------------------------
# File: src/dataset/utils.py
#
# Description:
#   This file contains utility classes and functions for handling the dataset
#   with PyTorch. It defines a custom Dataset to read the preprocessed data
#   and a collate class to handle batching of variable-length sequences.
#
# Usage:
#   This module is imported and used by the main trainer script.
# -----------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class BilingualDataset(Dataset):
    """
    A PyTorch Dataset for loading bilingual sentence pairs from tokenized text files.
    """
    def __init__(self, src_file, tgt_file, src_tokenizer, tgt_tokenizer):
        """
        Initializes the dataset.
        """
        super().__init__()

        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_lines = f.readlines()
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_lines = f.readlines()

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        # Get special token IDs as integers
        self.sos_id = self.tgt_tokenizer.token_to_id("[SOS]")
        self.eos_id = self.tgt_tokenizer.token_to_id("[EOS]")

    def __len__(self):
        """Returns the total number of sentence pairs."""
        return len(self.src_lines)

    def __getitem__(self, idx):
        """
        Retrieves a single sentence pair, converts to tensors, and adds special tokens.
        """
        src_line = self.src_lines[idx].strip()
        tgt_line = self.tgt_lines[idx].strip()

        # Convert space-separated token IDs back to integer lists
        src_ids = [int(id_) for id_ in src_line.split()] if src_line else []
        tgt_ids = [int(id_) for id_ in tgt_line.split()] if tgt_line else []

        # Prepare tensors for the model
        encoder_input = torch.tensor(src_ids, dtype=torch.long)
        decoder_input = torch.cat([torch.tensor([self.sos_id], dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)], dim=0)
        label = torch.cat([torch.tensor(tgt_ids, dtype=torch.long), torch.tensor([self.eos_id], dtype=torch.long)], dim=0)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "src_text": src_line,
            "tgt_text": tgt_line,
        }

class PadCollate:
    """
    A callable class to handle padding for batches of sentences.
    This is a more robust alternative to a static collate_fn.
    """
    def __init__(self, src_pad_id, tgt_pad_id):
        """
        Initializes the collator with the specific padding IDs.
        """
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

    def __call__(self, batch):
        """
        This method is called by the DataLoader to create a batch.
        """
        # Pad each part of the batch with its corresponding integer padding ID
        encoder_inputs = pad_sequence([item['encoder_input'] for item in batch], batch_first=True, padding_value=self.src_pad_id)
        decoder_inputs = pad_sequence([item['decoder_input'] for item in batch], batch_first=True, padding_value=self.tgt_pad_id)
        labels = pad_sequence([item['label'] for item in batch], batch_first=True, padding_value=self.tgt_pad_id)

        return {
            "encoder_input": encoder_inputs,
            "decoder_input": decoder_inputs,
            "label": labels
        }
