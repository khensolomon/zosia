# -----------------------------------------------------------------------------
# File: src/models/transformer.py
#
# Description:
#   This file assembles all the components (embeddings, attention, etc.)
#   to build the full Transformer model, including the Encoder and Decoder stacks.
#
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
from src.models.embeddings import InputEmbeddings, PositionalEncoding
from src.models.attention import MultiHeadAttention

# --- Building Blocks ---

class FeedForwardBlock(nn.Module):
    """
    Implements the Feed-Forward Network (FFN) sub-layer of the Transformer.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class EncoderBlock(nn.Module):
    """
    A single layer of the Transformer Encoder.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attention_output = self.self_attention_block(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward_block(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderBlock(nn.Module):
    """
    A single layer of the Transformer Decoder.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        self_attention_output = self.self_attention_block(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attention_output))
        cross_attention_output = self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attention_output))
        ff_output = self.feed_forward_block(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# --- Main Model Structure ---

class Encoder(nn.Module):
    """A stack of N EncoderBlocks."""
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(layers[0].self_attention_block.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    """A stack of N DecoderBlocks."""
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(layers[0].self_attention_block.d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    The final linear layer that projects the decoder output to the vocabulary size.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)

class Transformer(nn.Module):
    """
    The full Transformer model composed of an Encoder, a Decoder, and embeddings.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

# --- Factory Function ---

def build_transformer(src_vocab_size: int, tgt_vocab_size: int,
                      src_pad_idx: int, tgt_pad_idx: int,
                      d_model: int, num_encoder_layers: int, num_decoder_layers: int,
                      num_heads: int, d_ff: int, dropout: float, max_seq_len: int) -> Transformer:
    """A factory function to build the complete Transformer model."""
    # Create the embedding layers, now with padding_idx
    src_embed = InputEmbeddings(d_model, src_vocab_size, src_pad_idx)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size, tgt_pad_idx)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, max_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, max_seq_len, dropout)

    # Create the Encoder stack
    encoder_blocks = []
    for _ in range(num_encoder_layers):
        # FIX: Corrected typo from d_.model to d_model
        encoder_blocks.append(EncoderBlock(d_model, num_heads, d_ff, dropout))
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    # Create the Decoder stack
    decoder_blocks = []
    for _ in range(num_decoder_layers):
        decoder_blocks.append(DecoderBlock(d_model, num_heads, d_ff, dropout))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Assemble the Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize parameters with Xavier initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
