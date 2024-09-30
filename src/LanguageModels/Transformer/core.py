"""

Implements the Core Components of the Transformer Architecture

"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size, d_model):
    super(TokenEmbedding, self).__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
  
  def forward(self, x):
    return self.embedding(x)

class PositionalEncoding(nn.Module):
  def __init__(self, max_len, d_model):
    super(PositionalEncoding, self).__init__()
    self.encoding = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    self.encoding[:, 0::2] = torch.sin(position * div_term)
    self.encoding[:, 1::2] = torch.cos(position * div_term)
    self.encoding = self.encoding.unsqueeze(0)
  
  def forward(self, x):
    batch_size, seq_length, _ = x.size()
    encoding = self.encoding[:, :seq_length, :].expand(batch_size, -1, -1).to(x.device)
    return x + encoding

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_k = d_model // num_heads
    
    self.query = nn.Linear(d_model, d_model)
    self.key = nn.Linear(d_model, d_model)
    self.value = nn.Linear(d_model, d_model)
    self.fc_out = nn.Linear(d_model, d_model)
  
  def forward(self, q, k, v, mask=None):
    N = q.shape[0]
    
    q = self.query(q).view(N, -1, self.num_heads, self.d_k).transpose(1, 2)
    k = self.key(k).view(N, -1, self.num_heads, self.d_k).transpose(1, 2)
    v = self.value(v).view(N, -1, self.num_heads, self.d_k).transpose(1, 2)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e9)
    attention = torch.nn.functional.softmax(scores, dim=-1)
    
    x = torch.matmul(attention, v).transpose(1, 2).contiguous().view(N, -1, self.num_heads * self.d_k)
    return self.fc_out(x)

class FeedForwardNetwork(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1):
    super(FeedForwardNetwork, self).__init__()
    self.linear1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(d_ff, d_model)
  
  def forward(self, x):
    return self.linear2(self.dropout(F.relu(self.linear1(x))))

class LayerNormalization(nn.Module):
  def __init__(self, d_model, eps=1e-6):
    super(LayerNormalization, self).__init__()
    self.d_model = d_model
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(d_model))
    self.bias = nn.Parameter(torch.zeros(d_model))
  
  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias

class ResidualConnection(nn.Module):
  def __init__(self, d_model, dropout=0.1):
    super(ResidualConnection, self).__init__()
    self.layer_norm = LayerNormalization(d_model)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.layer_norm(x)))

class OutputLayer(nn.Module):
  def __init__(self, d_model, vocab_size):
    super(OutputLayer, self).__init__()
    self.linear = nn.Linear(d_model, vocab_size)
  
  def forward(self, x):
    return F.log_softmax(self.linear(x), dim=-1)
