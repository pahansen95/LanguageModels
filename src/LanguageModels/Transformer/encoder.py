"""

Implements the Encoder of the Transformer Architecture

"""
from __future__ import annotations

import torch.nn as nn

class EncoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
    super(EncoderLayer, self).__init__()
    self.self_attn = MultiHeadAttention(d_model, num_heads)
    self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
    self.residual1 = ResidualConnection(d_model, dropout)
    self.residual2 = ResidualConnection(d_model, dropout)

  def forward(self, x, mask=None):
    x = self.residual1(x, lambda x: self.self_attn(x, x, x, mask))
    return self.residual2(x, self.feed_forward)

class Encoder(nn.Module):
  def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1, max_len=512):
    super(Encoder, self).__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.positional_encoding = PositionalEncoding(max_len, d_model)
    self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
    self.layer_norm = nn.LayerNorm(d_model)

  def forward(self, src, src_mask=None):
    x = self.embedding(src) + self.positional_encoding(src)
    for layer in self.layers:
      x = layer(x, src_mask)
    return self.layer_norm(x)

### Relative Imports last to avoid Cyclic Imports
from .core import MultiHeadAttention, FeedForwardNetwork, ResidualConnection, PositionalEncoding
