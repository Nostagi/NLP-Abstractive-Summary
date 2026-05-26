import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy as copy

class SGNSTrainer(nn.Module):

    def __init__(self, embedding: nn.Embedding):
        super(SGNSTrainer, self).__init__()

        self.u_embedding = embedding
        self.v_embedding = copy(embedding)

    def forward(self, center: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        center:     [batch_size]
        positive:   [batch_size]
        negative:   [batch_size, negative_size]
        """

        center_emb = self.u_embedding(center)          # [batch_size, embed_dim]
        positive_emb = self.v_embedding(positive)      # [batch_size, embed_dim]
        negative_emb = self.v_embedding(negative)      # [batch_size, negative_size, embed_dim]

        pos_score = (positive_emb * center_emb).sum(dim=1)                       # [batch_size]
        neg_score = (negative_emb * center_emb.unsqueeze(1)).sum(dim=2)         # [batch_size, negative_size]

        pos_loss = F.logsigmoid(pos_score).squeeze()    # [batch_size]
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)  # [batch_size]

        total_loss = -(pos_loss + neg_loss).mean()  # [batch_size]

        return total_loss  
    
    
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

    def forward(self, seq_v: torch.Tensor) -> torch.Tensor:

        x = seq_v + self.pe[:, :seq_v.size(1)]

        return self.dropout(x)