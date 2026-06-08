import torch
import torch.nn as nn
import math

    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, dropout:float = 0.1, max_seq_len:int = 5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, d_model)

        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        # Thêm 1 chiều batch ở đầu để dễ dàng cộng với input tensor sau này
        # Shape: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, seq_v: torch.Tensor) -> torch.Tensor:

        x = seq_v * math.sqrt(self.d_model) + self.pe[:, :seq_v.size(1)]

        return self.dropout(x)