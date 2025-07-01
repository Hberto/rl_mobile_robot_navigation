import torch
import torch.nn as nn
import math
from .positional_encoding import PositionalEncoding


class HistoryEncoder(nn.Module):
    def __init__(self, state_dim, model_dim, history_size, dropout=0.1):
        super(HistoryEncoder, self).__init__()
        
        self.model_dim = model_dim
        self.embedding = nn.Linear(state_dim, model_dim)
        self.positional_encoder = PositionalEncoding(model_dim=model_dim, max_seq_length=history_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, history_batch):
        embedded_history = self.embedding(history_batch) * math.sqrt(self.model_dim)
        encoded_history = self.positional_encoder(embedded_history)
        
        return self.dropout(encoded_history)