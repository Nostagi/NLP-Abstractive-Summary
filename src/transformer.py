import math

import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy as copy

from .embedding import PositionalEncoding
from .interfaces import Network


class Transformer(Network):
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

        super(Transformer, self).__init__()

        # Attributes
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers    
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.seq_max_length: int = seq_max_len
        
        # Layers
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding: PositionalEncoding = PositionalEncoding(d_model, dropout=0.1, max_seq_len=seq_max_len)

        self.encoder: Encoder = Encoder(d_model, self.nhead, dim_feedforward, num_encoder_layers, dropout)
        self.decoder: Decoder = Decoder(d_model, self.nhead, dim_feedforward, num_decoder_layers, dropout)

        self.output_projection: nn.Linear = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.embedding.weight

    def make_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Tạo mask che đi các thẻ <PAD> của Encoder và Cross-Attention.

        Input: [batch_size, seq_len, *]
        Output: mask [batch_size, 1, 1, src_len]
        """
        # 1. Tạo boolean mask để đánh dấu vị trí của <PAD>
        is_pad = (src == self.pad_id)
    
        # 2. Tạo tensor chứa toàn số 0.0 với kiểu float
        mask = torch.zeros_like(src, dtype=torch.float)
        
        # 3. Fill -inf vào những vị trí is_pad == True
        mask = mask.masked_fill(is_pad, float('-inf'))
        
        # Reshape thành [batch_size, 1, 1, seq_len]
        mask = mask.unsqueeze(1).unsqueeze(2) 
        return mask

    def make_causal_mask(self, target_ids: torch.Tensor) -> torch.Tensor:

        seq_len = target_ids.size(1)

        # Tạo ma trận vuông [seq_len, seq_len] toàn 0
        mask = torch.zeros((seq_len, seq_len), device=target_ids.device)
        
        # Che phần phía trên đường chéo bằng âm vô cực
        # diagonal=1 nghĩa là che từ ngay trên đường chéo chính
        mask = mask.masked_fill(torch.triu(torch.ones((seq_len, seq_len), device=target_ids.device), diagonal=1).bool(), float('-inf'))
        
        # Mở rộng chiều để khớp với (batch_size, num_heads, seq_len, seq_len)
        mask = mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, seq_len]
        return mask

    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Input:
            source_ids: [batch_size, source_seq_len]
            target_ids: [batch_size, target_seq_len]

        Output:
            [batch_size, target_seq_len, vocab_size]
        """

        src_mask = self.make_padding_mask(source_ids)
        tgt_mask = self.make_causal_mask(target_ids)

        # 1. Embedding + Positional Encoding
        source_embedded = self.positional_encoding(self.embedding(source_ids))
        target_embedded = self.positional_encoding(self.embedding(target_ids))
        
        # 2. Encoder-Decoder
        enc_output = self.encoder(source_embedded, mask=src_mask)
        dec_output, _ = self.decoder(target_embedded, enc_output, source_mask=src_mask, target_mask=tgt_mask, cache=None)
        
        # 3. Output Projection
        logits = self.output_projection(dec_output)
        
        return logits
    
    def get_config(self) -> dict:
        """
        Trả về dictionary khớp 100% với tên các tham số trong hàm __init__.
        """
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "pad_id": self.pad_id,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "seq_max_len": self.seq_max_length
        }
    
    @torch.no_grad()
    def _decode_step(self, next_token: torch.Tensor, enc_output: torch.Tensor, 
                     src_mask: torch.Tensor, cache: list = None, step: int = 0):
        
        # TRICK: Khi dùng KV-Cache (chỉ nạp 1 token), KHÔNG cần dùng Causal Mask nữa!
        tgt_mask = None 

        # Đưa token duy nhất đi qua nhúng và Positional Encoding kết hợp với bước `step`
        target_embedded = self.positional_encoding(self.embedding(next_token), step=step)

        # Trả về cả logits và cache mới
        dec_output, new_cache = self.decoder(
            target_embedded, enc_output, 
            source_mask=src_mask, target_mask=tgt_mask, 
            cache=cache
        )

        logits = self.output_projection(dec_output) 
        return logits[:, -1, :], new_cache

    @torch.no_grad()
    def generate(self, source_ids: torch.Tensor, eos_id: int, device: torch.device = 'auto') -> torch.Tensor:
        """
        Hàm sinh tự hồi quy dựa trên kiến trúc forward chuẩn của mô hình.
        
        Input:
            source_ids: [batch_size, source_seq_len]
            bos_id: ID của thẻ <BOS>
            eos_id: ID của thẻ <EOS>
            max_len: Độ dài tối đa của văn bản sinh ra
            
        Output:
            target_ids: [batch_size, generated_seq_len]
        """

        self.eval()
        batch_size = source_ids.size(0)
        source_ids = source_ids.to(device)

        # 1. ENCODER
        src_mask = self.make_padding_mask(source_ids)
        source_embedded = self.positional_encoding(self.embedding(source_ids))
        enc_output = self.encoder(source_embedded, mask=src_mask)

        # 2. KHỞI TẠO
        next_token = torch.full((batch_size, 1), self.pad_id, dtype=torch.long, device=device)
        target_ids = next_token
        cache = None

        # 3. VÒNG LẶP SINH TỪ (Chỉ đưa next_token vào model, không đưa cả mảng target_ids)
        for step in range(self.seq_max_length):
            
            # Tính toán dựa trên token duy nhất và nối cache
            next_token_logits, cache = self._decode_step(next_token, enc_output, src_mask, cache, step)
            
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            target_ids = torch.cat([target_ids, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        return target_ids

    @torch.no_grad()
    def beam_search(self, source_ids: torch.Tensor, eos_id: int, beam_width: int = 3, device: torch.device = None) -> torch.Tensor:
        """
        Sinh văn bản bằng thuật toán Beam Search. Giữ lại Top-K nhánh tiềm năng nhất.
        Hỗ trợ batch_size = 1.
        """
        self.eval()
        source_ids = source_ids.to(device)

        src_mask = self.make_padding_mask(source_ids)
        source_embedded = self.positional_encoding(self.embedding(source_ids))
        enc_output = self.encoder(source_embedded, mask=src_mask)

        start_token = torch.full((1, 1), self.pad_id, dtype=torch.long, device=device)
        
        # MỖI BEAM LƯU 3 TRƯỜNG: (sequence, tổng điểm, cache của riêng nhánh đó)
        beams = [(start_token, 0.0, None)] 

        for step in range(self.seq_max_length):
            new_beams = []
            
            for seq, score, cache in beams:
                if seq[0, -1].item() == eos_id:
                    new_beams.append((seq, score, cache))
                    continue
                
                # CHỈ lấy duy nhất token cuối cùng của chuỗi để đưa vào dự đoán
                next_token = seq[:, -1:] 
                next_token_logits, new_cache = self._decode_step(next_token, enc_output, src_mask, cache, step)

                # --- Repetition Penalty ---
                generated_ids = seq[0].tolist()
                repetition_penalty = 1.5
                for token_id in set(generated_ids):
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty
                
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs[0], beam_width)
                
                for i in range(beam_width):
                    tok = topk_indices[i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, tok], dim=1)
                    new_score = score + topk_log_probs[i].item() 
                    
                    # Truyền cache mới sinh ra vào nhánh con
                    new_beams.append((new_seq, new_score, new_cache))
            
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            if all(seq[0, -1].item() == eos_id for seq, score, cache in beams):
                break

        best_seq = beams[0][0]
        return best_seq


# ------------------
# Components of the Transformer
# ------------------

class Encoder(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
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
                source_mask: torch.Tensor = None, target_mask: torch.Tensor = None,
                new_cache = []) -> torch.Tensor:
        
        if cache is None:
            cache = [None] * self.num_layers

        new_cache = []
        for layer in self.layers:
            x, layer_new_cache = layer(x, enc_output, source_mask, target_mask)
            new_cache.append(layer_new_cache)
            
        return self.norm(x), new_cache
    
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,) -> torch.Tensor:
        """
        Defines the forward pass of the Encoder block.
        Pre-Norm architecture.

        Input:
            [batch_size, seq_len]

        Output:
            [batch_size, seq_len, d_model]
        """
        x_norm = self.norm1(x)
        x_output, _ = self.self_attention(q=x_norm, k=x_norm, v=x_norm, mask=mask)
        x = x + self.dropout1(x_output)  # Residual connection + LayerNorm

        x_norm = self.norm2(x)
        x_output = self.feed_forward(x_norm)
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
                source_mask: torch.Tensor = None, target_mask: torch.Tensor = None,
                layer_cache: dict = None) -> torch.Tensor:

        if layer_cache is None:
            layer_cache = {'self': None, 'cross': None}

        x_norm = self.norm1(x)
        x_output, new_self_cache = self.self_attn(
            q=x_norm, k=x_norm, v=x_norm, 
            mask=target_mask, 
            kv_cache=layer_cache['self'], 
            is_cross_attn=False
        )
        x = x + self.dropout1(x_output)  # Residual connection + LayerNorm
        

        # Q: là x (thông tin từ Decoder)
        # K, V: là enc_output (thông tin từ Encoder)
        x_norm = self.norm2(x)
        x_output, new_cross_cache = self.cross_attn(
            q=x_norm, k=enc_output, v=enc_output, 
            mask=source_mask, 
            kv_cache=layer_cache['cross'], 
            is_cross_attn=True
        )
        x = x + self.dropout2(x_output)  # Residual connection + LayerNorm

        
        x_norm = self.norm3(x)
        x_output = self.ffn(x_norm)
        x = x + self.dropout3(x_output)  # Residual connection + LayerNorm
        
        new_cache = {'self': new_self_cache, 
                     'cross': new_cross_cache}


        return x, new_cache
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

        d_k = q.size(-1);

        # 1. Tính Q * K^T (Tích vô hướng)
        # Hàm .transpose(-2, -1) lật 2 chiều cuối cùng của ma trận K để có thể nhân ma trận
        # Kết quả: [batch, heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # 3. Masking (Che đi các phần tử không muốn mô hình nhìn thấy - <PAD>)
        if mask is not None:
            scores = scores + mask

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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None,
                kv_cache: tuple = None, 
                is_cross_attn: bool = False):
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

        if is_cross_attn and kv_cache is not None:
            K, V = kv_cache
        else:
            # Nếu chưa có, tính K và V
            K = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            
            # Nối tiếp (Concatenate) K, V hiện tại với K, V trong quá khứ (chỉ áp dụng cho Self-Attention)
            if not is_cross_attn and kv_cache is not None:
                K = torch.cat([kv_cache[0], K], dim=2)
                V = torch.cat([kv_cache[1], V], dim=2)
                
        # Cập nhật cache mới để trả ra ngoài
        new_cache = (K, V)
        
        # 3. Tính Attention trên nhiều heads (cùng lúc)
        scores, attention_weights = self.attention(Q, K, V, mask)
        
        # 4. GỘP HEADS LẠI
        # x đang có shape: [batch_size, num_heads, seq_len, d_k]
        # Bước A: transpose(1, 2) đưa về lại [batch_size, seq_len, num_heads, d_k]
        # Bước B: contiguous() là lệnh bắt buộc của PyTorch để sắp xếp lại bộ nhớ sau khi transpose
        # Bước C: view() gộp (num_heads * d_k) ngược trở lại thành d_model
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5. Đi qua lớp Linear cuối cùng để ra output
        # Final Shape: [batch_size, seq_len, d_model]
        output = self.w_o(concat)
        
        return output, new_cache