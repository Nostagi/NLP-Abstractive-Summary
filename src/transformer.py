import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy as copy

from .embedding import PositionalEncoding


# ------------------
# Components of the Transformer
# ------------------

class Encoder(nn.Module):

    def __init__(self, d_model: int, d_ff: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super(Encoder, self).__init__()
        encoder_layer = EncoderBlock(d_model, num_heads, d_ff, dropout)

        self.layers = nn.ModuleList([copy(encoder_layer) for _ in range(num_layers)])    
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        
        decoder_layer = DecoderBlock(d_model, num_heads, d_ff, dropout)
        
        self.layers = nn.ModuleList([copy(decoder_layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, 
                source_mask: torch.Tensor = None, target_mask: torch.Tensor = None) -> torch.Tensor:
        
        for layer in self.layers:
            x = layer(x, enc_output, source_mask, target_mask)
            
        return self.norm(x)
    
    
class Transformer(nn.Module):

    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 pad_id:int,
                 word_embedding: nn.Embedding, 
                 positional_encoding: PositionalEncoding,
                 encoder: Encoder, 
                 decoder: Decoder,
                 target_max_len:int = 1000):
        """
        Params:
            - `vocab_size` is the size of vocabulary (the one-hot vector as initial input).
            - `d_model` is the size of embedding vector (or `embed_dim`).
            - `pad_id` is the notation for <PAD>, will be used in sequence normalization.
            - `word_embedding`, `positional_encoding`, `encoder`, `decoder` are the transformer components that need to be pre-defined.
            - `target_max_len` is the expected maximum length of the model output
        """

        super(Transformer, self).__init__()

        # Attributes
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        
        # Layers
        self.embedding = word_embedding
        self.positional_encoding = positional_encoding

        self.encoder = encoder
        self.decoder = decoder

        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.embedding.weight

        # Others
        triag_mask = torch.tril(
            torch.ones((target_max_len, target_max_len), dtype=torch.bool)
        )

        self.register_buffer('triag_mask', triag_mask)

    def make_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Tạo mask che đi các thẻ <PAD> của Encoder và Cross-Attention.

        Input: [batch_size, seq_len, *]
        Output: mask [batch_size, 1, 1, src_len]
        """
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    def make_casual_mask(self, target: torch.Tensor) -> torch.Tensor:
        """
        Tạo mask cho Decoder (kết hợp Padding Mask và Causal Mask).

        Input : tgt [batch_size, target_len]
        Output: mask [batch_size, 1, target_len, target_len]
        """
        # 1. Pad Mask: [batch_size, 1, 1, tgt_len]
        pad_mask = self.make_padding_mask(target)

        len = pad_mask.size(-1)
        
        # 2. Causal Mask: [batch_size, 1, tgt_len, tgt_len]
        return pad_mask & self.triag_mask[:len, :len]

    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Input:
            source_ids: [batch_size, source_seq_len]
            target_ids: [batch_size, target_seq_len]

        Output:
            [batch_size, target_seq_len, vocab_size]
        """

        src_mask = self.make_padding_mask(source_ids)
        tgt_mask = self.make_casual_mask(target_ids)

        # 1. Embedding + Positional Encoding
        source_embedded = self.positional_encoding(self.embedding(source_ids))
        target_embedded = self.positional_encoding(self.embedding(target_ids))
        
        # 2. Encoder-Decoder
        enc_output = self.encoder(source_embedded, mask=src_mask)
        dec_output = self.decoder(target_embedded, enc_output, source_mask=src_mask, target_mask=tgt_mask)
        
        # 3. Output Projection
        logits = self.output_projection(dec_output)
        
        return logits

# ------------------
# Bigger blocks
# ------------------

class EncoderBlock(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super(EncoderBlock, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, nhead, dropout)

        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Defines the forward pass of the Encoder block.
        Pre-Norm architecture.

        Input:
            [batch_size, seq_len]

        Output:
            [batch_size, seq_len, d_model]
        """
        x = self.norm1(x)
        x_output = self.self_attention(q=x, k=x, v=x, mask=mask)
        x = x + self.dropout1(x_output)  # Residual connection + LayerNorm

        x = self.norm2(x)
        x_output = self.feed_forward(x)
        x = x + self.dropout2(x_output)  # Residual connection + LayerNorm

        return x

class DecoderBlock(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # 1. Masked Self-Attention (Cho câu đầu ra)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 2. Cross-Attention (Q lấy từ Decoder, K & V lấy từ Encoder)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 3. Feed Forward Network
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # 3 Lớp Norm và Dropout cho từng Sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, 
                source_mask: torch.Tensor = None, target_mask: torch.Tensor = None) -> torch.Tensor:

        x = self.norm1(x)
        x_output = self.self_attn(q=x, k=x, v=x, mask=target_mask)
        x_output = x + self.dropout1(x_output)  # Residual connection + LayerNorm
        

        # Q: là x (thông tin từ Decoder)
        # K, V: là enc_output (thông tin từ Encoder)
        x = self.norm2(x)
        x_output = self.cross_attn(q=x, k=enc_output, v=enc_output, mask=source_mask)
        x_output = x + self.dropout2(x_output)  # Residual connection + LayerNorm

        
        x = self.norm3(x)
        x_output = self.ffn(x)
        x_output = x + self.dropout3(x_output)  # Residual connection + LayerNorm
        
        return x

# ------------------
# Smaller layers (inside the bigger blocks)
# ------------------

class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        d_model: Số chiều của vector đầu vào (ví dụ: 512)
        d_ff: Số chiều của lớp ẩn (ví dụ: 2048)
        """
        super().__init__()
        # Lớp chiếu từ d_model lên d_ff (phóng to)
        self.linear_inside = nn.Linear(d_model, d_ff)
        
        # Hàm kích hoạt (Non-linearity)
        self.relu = nn.ReLU()
        
        # Lớp chiếu ngược từ d_ff về lại d_model (thu nhỏ)
        self.linear_outside = nn.Linear(d_ff, d_model)
        
        # Dropout rải rác tắt ngẫu nhiên các nơ-ron để chống học vẹt
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape:  [batch_size, seq_len, d_model]
        1st Linear:   [batch_size, seq_len, d_ff]
        2nd Linear:   [batch_size, seq_len, d_model] as final Output
        """
        x = self.linear_inside(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_outside(x)
        
        return x
    
class NormExample(nn.Module):
    """
    Ví dụ về Layer Normalization. Khi dùng thì thay bằng nn.LayerNorm.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        
        # gamma (Scale factor - mean)
        self.gamma = nn.Parameter(torch.ones(d_model))
        
        # beta (Shift factor - variance)
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape: [batch_size, seq_len, d_model]
        """
        # Tính mean trên chiều cuối cùng
        # keepdim=True để giữ nguyên số chiều (thay vì sập chiều cuối) để sau broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        
        # Tính phương sai (variance)
        # unbiased=False tương đương với chia cho N thay vì N-1
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Chuẩn hóa (Normalize) về chuẩn tắc (mean=0, var=1)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale và Shift theo distribution tốt hơn
        return self.gamma * x_norm + self.beta
    
class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        q, k, v shape: [batch_size, num_heads, seq_len, d_model]
        """

        d_k = k.size(-1);

        # 1. Tính Q * K^T (Tích vô hướng)
        # Hàm .transpose(-2, -1) lật 2 chiều cuối cùng của ma trận K để có thể nhân ma trận
        # Kết quả: [batch, heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float16))

        # 3. Masking (Che đi các phần tử không muốn mô hình nhìn thấy - <PAD>)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # qua softmax thì e^-1e9 ~ 0

        # 4. Softmax
        # Tính xác suất dọc theo chiều cuối cùng (dim=-1)
        attention_weights = F.softmax(scores, dim=-1)

        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v)  # [batch, heads, seq_len, d_model]

        return output, attention_weights
    
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Đảm bảo số chiều d_model có thể chia đều cho số lượng heads
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Số chiều của mỗi head
        
        # 1. Khai báo 3 lớp Linear Projector để concatinated Q_i, K_i, V_i từ dữ liệu gốc
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 2. Lớp Linear cuối cùng để gộp (concatenate) các heads lại
        self.w_o = nn.Linear(d_model, d_model)
        
        # Tái sử dụng single-head attention
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        batch_size = q.size(0)
        
        # 1. Đưa Q, K, V qua các lớp Linear Projector
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
        
        # 2. CHIA HEADS: Kỹ thuật quan trọng nhất trong PyTorch
        # Bước A: view() để chẻ d_model thành (num_heads * d_k)
        # Bước B: transpose(1, 2) để tráo đổi vị trí của seq_len và num_heads cho nhau
        #   -> Shape cuối: [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Tính Attention trên nhiều heads (cùng lúc)
        x, attention_weights = self.attention(Q, K, V, mask)
        
        # 4. GỘP HEADS LẠI
        # x đang có shape: [batch_size, num_heads, seq_len, d_k]
        # Bước A: transpose(1, 2) đưa về lại [batch_size, seq_len, num_heads, d_k]
        # Bước B: contiguous() là lệnh bắt buộc của PyTorch để sắp xếp lại bộ nhớ sau khi transpose
        # Bước C: view() gộp (num_heads * d_k) ngược trở lại thành d_model
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5. Đi qua lớp Linear cuối cùng để ra output
        # Final Shape: [batch_size, seq_len, d_model]
        output = self.w_o(x)
        
        return output