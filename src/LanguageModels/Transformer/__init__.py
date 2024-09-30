"""

Implements the Transformer Architecture

"""
from __future__ import annotations
import torch.nn as nn

class Transformer(nn.Module):
  def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1, max_len=512):
    super(Transformer, self).__init__()
    self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
    self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
    self.output_layer = OutputLayer(d_model, tgt_vocab_size)

  def forward(self, src, tgt, src_mask=None, tgt_mask=None):
    enc_output = self.encoder(src, src_mask)
    dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
    return self.output_layer(dec_output)

### Relative Imports last to avoid Cyclic Imports
from .encoder import Encoder
from .decoder import Decoder
from .core import OutputLayer