# Token and Positional Embeddings for the Transformer.

import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    """
    Standard token embedding layer.
    """
    def __init__(self, vocab_size: int, hid_dim: int):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            hid_dim (int): Dimension of the embedding vectors.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hid_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Input tensor of token IDs (batch_size, seq_len).

        Returns:
            torch.Tensor: Embedded tokens (batch_size, seq_len, hid_dim).
        """
        return self.embedding(src)

class PositionalEncoding(nn.Module):
    """
    Positional Encoding as described in "Attention Is All You Need".
    Adds positional information to token embeddings.
    """
    def __init__(self, hid_dim: int, max_seq_len: int, dropout: float, device: torch.device):
        """
        Args:
            hid_dim (int): Dimension of the embedding vectors.
            max_seq_len (int): Maximum sequence length the model expects.
            dropout (float): Dropout probability.
            device (torch.device): Device to run the computations on.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Create positional encoding matrix (max_seq_len, hid_dim)
        pe = torch.zeros(max_seq_len, hid_dim).to(device)

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, hid_dim, 2).float() * (-math.log(10000.0) / hid_dim)).to(device)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, max_seq_len, hid_dim) for broadcasting
        self.register_buffer('pe', pe) # Register as buffer so it's part of state_dict but not trained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to input embeddings.

        Args:
            x (torch.Tensor): Input token embeddings (batch_size, seq_len, hid_dim).

        Returns:
            torch.Tensor: Embeddings with positional information (batch_size, seq_len, hid_dim).
        """
        seq_len = x.shape[1]
        # Positional encoding is added, scaled by sqrt(hid_dim) as per paper (implicitly by adding it to scaled embeddings)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)