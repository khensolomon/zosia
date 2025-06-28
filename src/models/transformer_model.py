# The main Transformer Encoder-Decoder architecture.

import torch
import torch.nn as nn

from src.models.attention import MultiHeadAttention
from src.models.embeddings import TokenEmbedding, PositionalEncoding

class PositionwiseFeedforwardLayer(nn.Module):
    """
    Position-wise Feedforward Network used in Transformer.
    """
    def __init__(self, hid_dim: int, pf_dim: int, dropout: float):
        """
        Args:
            hid_dim (int): Input and output dimension.
            pf_dim (int): Inner dimension of the feedforward layer.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, hid_dim).

        Returns:
            torch.Tensor: Output tensor (batch_size, seq_len, hid_dim).
        """
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

class EncoderLayer(nn.Module):
    """
    Single Encoder Layer of the Transformer.
    Consists of Multi-Head Attention and Position-wise Feedforward Network.
    """
    def __init__(self, hid_dim: int, n_heads: int, pf_dim: int, dropout: float, device: torch.device):
        """
        Args:
            hid_dim (int): Dimension of hidden states.
            n_heads (int): Number of attention heads.
            pf_dim (int): Dimension of the position-wise feedforward network.
            dropout (float): Dropout probability.
            device (torch.device): Device to run the computations on.
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Input source sequence (batch_size, src_len, hid_dim).
            src_mask (torch.Tensor): Mask for source sequence (batch_size, 1, 1, src_len).

        Returns:
            torch.Tensor: Output of the encoder layer (batch_size, src_len, hid_dim).
        """
        # Self-attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src)) # Add & Norm

        # Position-wise Feedforward
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src)) # Add & Norm

        return src

class Encoder(nn.Module):
    """
    Transformer Encoder.
    Comprises token and positional embeddings, and multiple EncoderLayers.
    """
    def __init__(self,
                 input_dim: int, # Source vocabulary size
                 hid_dim: int,
                 n_layers: int,
                 n_heads: int,
                 pf_dim: int,
                 dropout: float,
                 max_seq_len: int,
                 device: torch.device):
        """
        Args:
            input_dim (int): Size of the source vocabulary.
            hid_dim (int): Dimension of hidden states.
            n_layers (int): Number of encoder layers.
            n_heads (int): Number of attention heads.
            pf_dim (int): Dimension of the position-wise feedforward network.
            dropout (float): Dropout probability.
            max_seq_len (int): Maximum sequence length.
            device (torch.device): Device to run the computations on.
        """
        super().__init__()
        self.device = device
        self.token_embedding = TokenEmbedding(input_dim, hid_dim)
        self.pos_embedding = PositionalEncoding(hid_dim, max_seq_len, dropout, device)
        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Input source sequence (batch_size, src_len).
            src_mask (torch.Tensor): Mask for source sequence (batch_size, 1, 1, src_len).

        Returns:
            torch.Tensor: Encoded source representation (batch_size, src_len, hid_dim).
        """
        # (batch_size, src_len) -> (batch_size, src_len, hid_dim)
        src = self.dropout(self.token_embedding(src) * self.scale)
        src = self.pos_embedding(src)

        for layer in self.layers:
            src = layer(src, src_mask)

        return src

class DecoderLayer(nn.Module):
    """
    Single Decoder Layer of the Transformer.
    Consists of masked Multi-Head Self-Attention, Encoder-Decoder Attention,
    and Position-wise Feedforward Network.
    """
    def __init__(self, hid_dim: int, n_heads: int, pf_dim: int, dropout: float, device: torch.device):
        """
        Args:
            hid_dim (int): Dimension of hidden states.
            n_heads (int): Number of attention heads.
            pf_dim (int): Dimension of the position-wise feedforward network.
            dropout (float): Dropout probability.
            device (torch.device): Device to run the computations on.
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.encoder_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                trg: torch.Tensor,
                enc_src: torch.Tensor,
                trg_mask: torch.Tensor,
                src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trg (torch.Tensor): Input target sequence (batch_size, trg_len, hid_dim).
            enc_src (torch.Tensor): Encoded source representation from Encoder (batch_size, src_len, hid_dim).
            trg_mask (torch.Tensor): Mask for target sequence (batch_size, 1, trg_len, trg_len).
            src_mask (torch.Tensor): Mask for source sequence (batch_size, 1, 1, src_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Output of the decoder layer (batch_size, trg_len, hid_dim).
                - Self-attention weights.
                - Encoder-Decoder attention weights.
        """
        # Self-attention (masked to prevent attending to future tokens)
        _trg, self_attention = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # Encoder-Decoder attention (queries from target, keys/values from encoded source)
        _trg, encoder_attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # Position-wise Feedforward
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, self_attention, encoder_attention

class Decoder(nn.Module):
    """
    Transformer Decoder.
    Comprises token and positional embeddings, multiple DecoderLayers, and a final linear layer for vocabulary prediction.
    """
    def __init__(self,
                 output_dim: int, # Target vocabulary size
                 hid_dim: int,
                 n_layers: int,
                 n_heads: int,
                 pf_dim: int,
                 dropout: float,
                 max_seq_len: int,
                 device: torch.device):
        """
        Args:
            output_dim (int): Size of the target vocabulary.
            hid_dim (int): Dimension of hidden states.
            n_layers (int): Number of decoder layers.
            n_heads (int): Number of attention heads.
            pf_dim (int): Dimension of the position-wise feedforward network.
            dropout (float): Dropout probability.
            max_seq_len (int): Maximum sequence length.
            device (torch.device): Device to run the computations on.
        """
        super().__init__()
        self.device = device
        self.token_embedding = TokenEmbedding(output_dim, hid_dim)
        self.pos_embedding = PositionalEncoding(hid_dim, max_seq_len, dropout, device)
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self,
                trg: torch.Tensor,
                enc_src: torch.Tensor,
                trg_mask: torch.Tensor,
                src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trg (torch.Tensor): Input target sequence (batch_size, trg_len).
            enc_src (torch.Tensor): Encoded source representation (batch_size, src_len, hid_dim).
            trg_mask (torch.Tensor): Mask for target sequence (batch_size, 1, trg_len, trg_len).
            src_mask (torch.Tensor): Mask for source sequence (batch_size, 1, 1, src_len).

        Returns:
            torch.Tensor: Predicted logits over target vocabulary (batch_size, trg_len, output_dim).
        """
        # (batch_size, trg_len) -> (batch_size, trg_len, hid_dim)
        trg = self.dropout(self.token_embedding(trg) * self.scale)
        trg = self.pos_embedding(trg)

        for layer in self.layers:
            trg, _, _ = layer(trg, enc_src, trg_mask, src_mask)

        # (batch_size, trg_len, hid_dim) -> (batch_size, trg_len, output_dim)
        output = self.fc_out(trg)

        return output

class Transformer(nn.Module):
    """
    The full Transformer Encoder-Decoder model.
    """
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_pad_idx: int,
                 trg_pad_idx: int,
                 device: torch.device):
        """
        Args:
            encoder (Encoder): The Transformer encoder.
            decoder (Decoder): The Transformer decoder.
            src_pad_idx (int): Padding token ID for source sequences.
            trg_pad_idx (int): Padding token ID for target sequences.
            device (torch.device): Device to run the computations on.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Creates a mask for source sequence padding.
        (batch_size, src_len) -> (batch_size, 1, 1, src_len)
        """
        # Mask is 1 where not pad, 0 where pad
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """
        Creates a combined mask for target sequence:
        1. Padding mask: Masks out padding tokens.
        2. Subsequent mask: Prevents attention to future tokens during decoding.
        (batch_size, trg_len) -> (batch_size, 1, trg_len, trg_len)
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Source sequence (batch_size, src_len).
            trg (torch.Tensor): Target sequence (batch_size, trg_len).

        Returns:
            torch.Tensor: Predicted logits over target vocabulary (batch_size, trg_len, output_dim).
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output