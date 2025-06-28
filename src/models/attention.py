# Implementation of Multi-Head Attention.

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as described in "Attention Is All You Need".
    """
    def __init__(self, hid_dim: int, n_heads: int, dropout: float, device: torch.device):
        """
        Args:
            hid_dim (int): Dimension of the input and output.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            device (torch.device): Device to run the computations on (CPU or CUDA).
        """
        super().__init__()

        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads"

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads # Dimension of each head

        # Linear transformations for Q, K, V
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim) # Output linear layer

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass for Multi-Head Attention.

        Args:
            query (torch.Tensor): Query tensor (batch_size, query_len, hid_dim).
            key (torch.Tensor): Key tensor (batch_size, key_len, hid_dim).
            value (torch.Tensor): Value tensor (batch_size, value_len, hid_dim).
            mask (torch.Tensor, optional): Mask tensor (batch_size, 1, 1, key_len or query_len).
                                         Used to mask out padding or future tokens. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output tensor after attention (batch_size, query_len, hid_dim).
                - Attention weights (batch_size, n_heads, query_len, key_len).
        """
        batch_size = query.shape[0]

        # Apply linear transformations
        # (batch_size, seq_len, hid_dim) -> (batch_size, seq_len, hid_dim)
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Reshape to (batch_size, n_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate attention scores
        # (batch_size, n_heads, query_len, head_dim) @ (batch_size, n_heads, head_dim, key_len)
        # -> (batch_size, n_heads, query_len, key_len)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            # Apply mask by filling masked positions with a very small negative number
            # which becomes ~0 after softmax
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1) # (batch_size, n_heads, query_len, key_len)

        x = torch.matmul(self.dropout(attention), V) # (batch_size, n_heads, query_len, head_dim)

        # Reshape back to original hid_dim and concatenate heads
        # (batch_size, query_len, n_heads, head_dim) -> (batch_size, query_len, hid_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)

        # Apply final linear layer
        x = self.fc_o(x) # (batch_size, query_len, hid_dim)

        return x, attention