# -----------------------------------------------------------------------------
# File: src/models/attention.py
#
# Description:
#   This file implements the Multi-Head Attention mechanism, a core component
#   of the Transformer model.
#
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism as described in the
    "Attention Is All You Need" paper.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model) # Query weight matrix
        self.w_k = nn.Linear(d_model, d_model) # Key weight matrix
        self.w_v = nn.Linear(d_model, d_model) # Value weight matrix
        self.w_o = nn.Linear(d_model, d_model) # Output weight matrix
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        The core scaled dot-product attention function.
        """
        d_k = query.shape[-1]
        
        # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # Apply the mask by setting masked positions to a very small number.
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) -> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # 1. Apply linear layers to get Q, K, V
        # Input shape: (batch, seq_len, d_model)
        query = self.w_q(q) # (batch, seq_len, d_model)
        key = self.w_k(k)   # (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model)

        # 2. Reshape Q, K, V for multi-head processing
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        # 3. Compute attention
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        
        # 4. Concatenate heads and apply final linear layer
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)
