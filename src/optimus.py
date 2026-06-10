from .embedding import RotaryPositionalEmbedding, PositionalEncodingBypass
from .transformer import ScaledDotProductAttention
from . import transformer as traditional

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as copy


class Transformer(traditional.Transformer):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 pad_id:int,
                 nhead: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 seq_max_len:int = 1000):
        """
        Params:
            - `vocab_size` is the size of vocabulary (the one-hot vector as initial input).
            - `d_model` is the size of embedding vector (or `embed_dim`).
            - `pad_id` is the notation for <PAD>, will be used in sequence normalization.
            - `word_embedding`, `positional_encoding`, `encoder`, `decoder` are the transformer components that need to be pre-defined.
            - `seq_max_len` is the expected maximum length of the input and output sequences
        """

        super().__init__(vocab_size, d_model, pad_id, nhead, num_encoder_layers, 
                         num_decoder_layers, dim_feedforward, dropout, seq_max_len)
        
        self.encoder = Encoder(d_model, nhead, dim_feedforward, num_encoder_layers, dropout)
        self.decoder = Decoder(d_model, nhead, dim_feedforward, num_decoder_layers, dropout)

        # Đánh dấu Bypass cho lớp PE 
        self.positional_encoding.set_bypass(True)


    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """ Ghi đè để hứng đúng Tuple từ Decoder khi Training """
        src_mask = self.make_padding_mask(source_ids)
        tgt_mask = self.make_causal_mask(target_ids)

        source_embedded = self.positional_encoding(self.embedding(source_ids))
        target_embedded = self.positional_encoding(self.embedding(target_ids))
        
        enc_output = self.encoder(source_embedded, mask=src_mask)
        
        # Hứng dec_output, vứt bỏ cache (_)
        dec_output, _ = self.decoder(
            target_embedded, enc_output, 
            source_mask=src_mask, target_mask=tgt_mask,
            cache=None, step=0
        )
        
        logits = self.output_projection(dec_output)
        return logits
    
    @torch.no_grad()
    def _decode_step(self, next_token: torch.Tensor, enc_output: torch.Tensor, 
                     src_mask: torch.Tensor, cache: list = None, step: int = 0):
        
        tgt_mask = None 
        target_embedded = self.positional_encoding(self.embedding(next_token), step=step)

        # QUAN TRỌNG: Phải ép Decoder nhận biến step để truyền xuống RoPE
        dec_output, new_cache = self.decoder(
            target_embedded, enc_output, 
            source_mask=src_mask, target_mask=tgt_mask, 
            cache=cache, 
            step=step
        )

        logits = self.output_projection(dec_output) 
        return logits[:, -1, :], new_cache
    

# ------------------
# Components of the Transformer
# ------------------

class Encoder(traditional.Encoder):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
        super().__init__(d_model, num_heads, d_ff, num_layers, dropout)

        # Ghi đè kiến trúc mới
        encoder_layer = EncoderBlock(d_model, num_heads, d_ff, dropout)
        self.layers = nn.ModuleList([copy(encoder_layer) for _ in range(num_layers)])  

        self.norm = RMSNorm(d_model)

class Decoder(traditional.Decoder):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
        super().__init__(d_model, num_heads, d_ff, num_layers, dropout)
        
        # Ghi đè kiến trúc mới
        decoder_layer = DecoderBlock(d_model, num_heads, d_ff, dropout)
        self.layers = nn.ModuleList([copy(decoder_layer) for _ in range(num_layers)])

        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, 
                source_mask: torch.Tensor = None, target_mask: torch.Tensor = None,
                cache: list = None, step: int = 0):
        
        if cache is None:
            cache = [None] * self.num_layers
            
        new_cache = []
        for i, layer in enumerate(self.layers):
            # Truyền step xuống từng layer
            x, layer_new_cache = layer(x, enc_output, source_mask, target_mask, cache[i], step)
            new_cache.append(layer_new_cache)
            
        return self.norm(x), new_cache

# ------------------
# Bigger blocks
# ------------------

class EncoderBlock(traditional.EncoderBlock):

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__(d_model, nhead, d_ff, dropout)
        
        # 1. Đè Attention bằng RoPE
        self.self_attention = RoPEMultiHeadAttention(d_model, nhead, dropout)
        
        # 2. Đè các lớp LayerNorm bằng RMSNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # 3. Đè lớp Fully Connected Feed Forward
        self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout)

class DecoderBlock(traditional.DecoderBlock):
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__(d_model, num_heads, d_ff, dropout)
        
        # 1. Đè Self-Attention bằng RoPE. Cross-Attention giữ nguyên.
        self.self_attn = RoPEMultiHeadAttention(d_model, num_heads, dropout)
        
        # 2. Đè 3 lớp LayerNorm bằng RMSNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

        # 3. Đè lớp Fully Connected Feed Forward
        self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, 
                source_mask: torch.Tensor = None, target_mask: torch.Tensor = None,
                layer_cache: dict = None, step: int = 0):
        
        if layer_cache is None:
            layer_cache = {'self': None, 'cross': None}

        x_norm = self.norm1(x)
        # Truyền step vào cho RoPE
        x_output, new_self_cache = self.self_attn(
            q=x_norm, k=x_norm, v=x_norm, 
            mask=target_mask, 
            kv_cache=layer_cache['self'], 
            is_cross_attn=False,
            step=step 
        )
        x = x + self.dropout1(x_output)  

        x_norm = self.norm2(x)
        x_output, new_cross_cache = self.cross_attn(
            q=x_norm, k=enc_output, v=enc_output, 
            mask=source_mask, 
            kv_cache=layer_cache['cross'], 
            is_cross_attn=True
        )
        x = x + self.dropout2(x_output)  

        x_norm = self.norm3(x)
        x_output = self.ffn(x_norm)
        x = x + self.dropout3(x_output)  
        
        new_cache = {'self': new_self_cache, 'cross': new_cross_cache}
        return x, new_cache

# ------------------
# Smaller layers (inside the bigger blocks)
# ------------------

class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads 
        
        # 1. Các lớp Linear Projector
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # 2. Module RoPE: Khởi tạo với số chiều là d_k (head_dim)
        self.rope = RotaryPositionalEmbedding(head_dim=self.d_k, max_seq_len=max_seq_len)
        
        # 3. Module Scaled Dot-Product Attention (tái sử dụng từ code gốc)
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None,
                kv_cache: tuple = None, 
                is_cross_attn: bool = False, step: int = 0):
        """
        Input (q, k, v shape): [batch_size, seq_len, d_model]
        Output shape: [batch_size, seq_len, d_model]
        """
        batch_size = q.size(0)
        
        # 1. Linear Projections
        # Shape: [batch_size, seq_len, d_model]
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
        
        # 2. Chia Heads
        # Shape sau transpose: [batch_size, num_heads, seq_len, d_k]
        batch_size = q.size(0)
        
        # Luôn tính Q mới
        Q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        if is_cross_attn and kv_cache is not None:
            K, V = kv_cache
        else:
            K = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            
            # 1. XOAY RoPE (Truyền step vào để xoay đúng vị trí)
            Q, K = self.rope(Q, K, step=step)
            
            # 2. NỐI CACHE (Sau khi đã xoay xong K mới)
            if not is_cross_attn and kv_cache is not None:
                K = torch.cat([kv_cache[0], K], dim=2)
                V = torch.cat([kv_cache[1], V], dim=2)
                
        new_cache = (K, V)
        
        # 4. Tính Attention trên nhiều heads
        # Output shape từ attention: [batch_size, num_heads, seq_len, d_k]
        scores, attention_weights = self.attention(Q, K, V, mask)
        
        # 5. Gộp Heads (Concatenate)
        # Shape: [batch_size, seq_len, d_model]
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 6. Linear Output
        # Final Shape: [batch_size, seq_len, d_model]
        output = self.w_o(concat)
        
        return output, new_cache

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Khởi tạo RMSNorm.
        Lưu ý: Không có bias (beta), chỉ có weight (gamma).
        """
        super().__init__()
        self.eps = eps
        # Tham số gamma học được, khởi tạo bằng 1
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Tính Root Mean Square (RMS) và chia x cho RMS
        # pow(2): Bình phương
        # mean(-1): Tính trung bình trên chiều cuối cùng (d_model)
        # rsqrt: 1 / căn bậc 2
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trick thực chiến: Ép kiểu x sang float32 để tính Norm tránh sai số, 
        # sau đó trả về kiểu gốc của x (type_as)
        output = self._norm(x.float()).type_as(x)
        
        # Nhân với tham số học được gamma
        return output * self.weight
    
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        
        # Nếu không cung cấp d_ff, tự động tính toán theo công thức 8/3 của bài báo
        if d_ff is None:
            hidden_dim = int(8 * d_model / 3)
            # Thường trong thực tế sẽ làm tròn hidden_dim để chia hết cho 256/128 nhằm tối ưu GPU
        else:
            hidden_dim = d_ff
            
        # 3 Lớp Linear (không dùng bias)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False) # Tạo Gate
        self.v = nn.Linear(d_model, hidden_dim, bias=False)  # Tạo Value
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False) # Down projection
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [batch_size, seq_len, d_model]
        Output: [batch_size, seq_len, d_model]
        """
        # 1. Luồng Gate đi qua hàm SiLU (Swish với beta=1)
        gate = F.silu(self.w1(x))
        
        # 2. Luồng Value (chỉ chiếu tuyến tính, không qua kích hoạt)
        value = self.v(x)
        
        # 3. Nhân Element-wise hai luồng
        hidden = gate * value
        
        # 4. Chiếu ngược về d_model và Dropout
        output = self.dropout(self.w2(hidden))
        
        return output