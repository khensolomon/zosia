# -----------------------------------------------------------------------------
# File: src/models/embeddings.py
#
# Description:
#   This file contains the implementation of the input embeddings and the
#   positional encoding for the Transformer model.
#
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    Converts input token IDs into dense vector representations (embeddings).
    It now accepts a padding_idx to handle padding tokens correctly.
    """
    def __init__(self, d_model: int, vocab_size: int, pad_idx: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        # Tell the embedding layer which index is for padding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

    def forward(self, x):
        # The paper scales the embeddings by sqrt(d_model).
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings so that they can be summed.
    """
    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a positional encoding matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        
        # Create a tensor representing positions (0, 1, 2, ..., max_seq_len-1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # The formula for PE is based on sine and cosine functions of different frequencies.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension so it can be added to the input embeddings.
        pe = pe.unsqueeze(0) # Shape: (1, max_seq_len, d_model)

        # Register 'pe' as a buffer.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
