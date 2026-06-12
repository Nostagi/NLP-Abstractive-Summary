import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as copy

from . import transformer as traditional
from .optimus import RoPEMultiHeadAttention, RMSNorm, SwiGLUFeedForward

class GPTTransformer(traditional.Transformer):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 pad_id: int,
                 nhead: int,
                 num_layers: int,       
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 seq_max_len: int = 2048):
        
        # Khởi tạo cha. Khai báo đúng các tham số để duy trì get_config() chính xác
        super().__init__(vocab_size, d_model, pad_id, nhead, 0, num_layers, dim_feedforward, dropout, seq_max_len)
        
        # Dọn dẹp module Encoder/Decoder cũ từ traditional.Transformer
        del self.encoder
        del self.decoder
        
        # Tắt Positional Encoding truyền thống vì OptimusDecoderOnly đã dùng RoPE
        self.positional_encoding.set_bypass(True)
        
        # Gắn khối Decoder-Only (kết hợp các thành phần từ optimus.py)
        self.decoder = DecoderOnly(d_model, nhead, dim_feedforward, num_layers, dropout)

    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Ghi đè để tương thích ngược. 
        Tự động nối chuỗi source và target bên trong mô hình.
        """
        # 1. Cắt bỏ thẻ <EOS> ở cuối source_ids (Giả định token cuối là EOS hoặc SEP)
        # Bằng cách lấy từ đầu đến sát phần tử cuối cùng [:, :-1]
        source_ids_stripped = source_ids[:, :-1]
        
        # 2. Concat phần còn lại với target_ids thành 1 chuỗi input_ids duy nhất
        input_ids = torch.cat([source_ids_stripped, target_ids], dim=1)
        
        # 3. Tạo combined mask trên chuỗi gộp
        pad_mask = self.make_padding_mask(input_ids)
        causal_mask = self.make_causal_mask(input_ids)
        combined_mask = torch.maximum(pad_mask, causal_mask)
        
        # 4. Truyền qua mạng (step=0 cho Prefill toàn bộ chuỗi)
        embedded = self.positional_encoding(self.embedding(input_ids))
        dec_output, _ = self.decoder(embedded, mask=combined_mask, cache=None, step=0)
        
        # 5. Chiếu ra Vocab
        logits = self.output_projection(dec_output)
        return logits

    @torch.no_grad()
    def _decode_step(self, next_token: torch.Tensor, cache: list = None, step: int = 0):
        """
        Bước nhảy cho 1 token duy nhất với KV-Cache. Mask = None vì không nhìn về tương lai.
        """
        embedded = self.positional_encoding(self.embedding(next_token), step=step)
        
        dec_output, new_cache = self.decoder(
            embedded, mask=None, cache=cache, step=step
        )
        
        logits = self.output_projection(dec_output) 
        return logits[:, -1, :], new_cache

    @torch.no_grad()
    def generate(self, source_ids: torch.Tensor, eos_id: int, max_new_tokens: int = None, device: torch.device = 'cpu') -> torch.Tensor:
        self.eval()
        source_ids = source_ids.to(device)

        if not max_new_tokens :
            max_new_tokens = self.seq_max_length

        # Xử lý an toàn: Nếu source_ids có chứa <EOS> ở cuối, ta cắt nó đi trước khi làm prompt
        if source_ids.size(1) > 0 and (source_ids[:, -1] == eos_id).all():
            source_ids = source_ids[:, :-1]

        seq_len = source_ids.size(1)

        # ==========================================
        # PHASE 1: PREFILL (Đọc hiểu Prompt gốc)
        # ==========================================
        pad_mask = self.make_padding_mask(source_ids)
        causal_mask = self.make_causal_mask(source_ids)
        combined_mask = torch.maximum(pad_mask, causal_mask)
        
        embedded = self.positional_encoding(self.embedding(source_ids))
        dec_output, cache = self.decoder(embedded, mask=combined_mask, cache=None, step=0)
        
        logits = self.output_projection(dec_output)
        next_token_logits = logits[:, -1, :] 
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        generated_ids = torch.cat([source_ids, next_token], dim=1)

        # ==========================================
        # PHASE 2: DECODE (Sinh hồi quy với KV Cache)
        # ==========================================
        for step in range(seq_len, seq_len + max_new_tokens):
            if (next_token == eos_id).all():
                break
            
            next_token_logits, cache = self._decode_step(next_token, cache=cache, step=step)
            
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

        return generated_ids[:, seq_len:]

    @torch.no_grad()
    def beam_search(self, source_ids: torch.Tensor, eos_id: int, beam_width: int = 3, max_new_tokens: int = None, device: torch.device = 'cpu') -> torch.Tensor:
        self.eval()
        source_ids = source_ids.to(device)

        if not max_new_tokens :
            max_new_tokens = self.seq_max_length
        
        # Xử lý an toàn: Cắt <EOS> ở cuối source_ids tương tự generate
        if source_ids.size(1) > 0 and (source_ids[:, -1] == eos_id).all():
            source_ids = source_ids[:, :-1]
            
        seq_len = source_ids.size(1)
        
        def clone_cache(c):
            if c is None: return None
            return [(k.clone(), v.clone()) if k is not None else None for k, v in c]

        # --- PREFILL ---
        pad_mask = self.make_padding_mask(source_ids)
        causal_mask = self.make_causal_mask(source_ids)
        combined_mask = torch.maximum(pad_mask, causal_mask)
        
        embedded = self.positional_encoding(self.embedding(source_ids))
        dec_output, cache = self.decoder(embedded, mask=combined_mask, cache=None, step=0)
        
        logits = self.output_projection(dec_output)
        next_token_logits = logits[:, -1, :]
        
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        topk_log_probs, topk_indices = torch.topk(log_probs[0], beam_width)
        
        beams = []
        for i in range(beam_width):
            tok = topk_indices[i].unsqueeze(0).unsqueeze(0)
            seq = torch.cat([source_ids, tok], dim=1)
            score = topk_log_probs[i].item()
            beams.append((seq, score, clone_cache(cache)))
            
        # --- DECODE ---
        for step in range(seq_len, seq_len + max_new_tokens):
            new_beams = []
            
            for seq, score, b_cache in beams:
                if seq[0, -1].item() == eos_id:
                    new_beams.append((seq, score, b_cache))
                    continue
                    
                next_token = seq[:, -1:]
                step_logits, new_cache = self._decode_step(next_token, cache=b_cache, step=step)
                
                # Penalty lặp từ
                repetition_penalty = 1.2
                generated_list = seq[0].tolist()
                for token_id in set(generated_list):
                    if step_logits[0, token_id] < 0:
                        step_logits[0, token_id] *= repetition_penalty
                    else:
                        step_logits[0, token_id] /= repetition_penalty
                        
                step_log_probs = F.log_softmax(step_logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(step_log_probs[0], beam_width)
                
                for i in range(beam_width):
                    tok = topk_indices[i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, tok], dim=1)
                    new_score = score + topk_log_probs[i].item()
                    new_beams.append((new_seq, new_score, clone_cache(new_cache)))
                    
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(seq[0, -1].item() == eos_id for seq, score, c in beams):
                break
                
        best_seq = beams[0][0]
        return best_seq[:, seq_len:]


class DecoderOnly(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        
        decoder_layer = DecoderOnlyBlock(d_model, num_heads, d_ff, dropout)
        self.layers = nn.ModuleList([copy(decoder_layer) for _ in range(num_layers)])
        
        # Final Norm của cả chuỗi
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, 
                cache: list = None, step: int = 0) -> tuple:
        if cache is None:
            cache = [None] * self.num_layers
            
        new_cache = []
        for i, layer in enumerate(self.layers):
            x, layer_new_cache = layer(x, mask, cache[i], step)
            new_cache.append(layer_new_cache)
            
        return self.norm(x), new_cache


class DecoderOnlyBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Chỉ có Self-Attention (sử dụng RoPE)
        self.self_attn = RoPEMultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed Forward sử dụng SwiGLU
        self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout)
        
        # Pre-Norm với RMSNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, 
                layer_cache: tuple = None, step: int = 0) -> tuple:
        
        x_norm = self.norm1(x)
        
        # Truyền step vào để xoay RoPE chính xác
        x_output, new_cache = self.self_attn(
            q=x_norm, k=x_norm, v=x_norm, 
            mask=mask, 
            kv_cache=layer_cache, 
            is_cross_attn=False,
            step=step
        )
        x = x + self.dropout1(x_output)

        x_norm = self.norm2(x)
        x_output = self.ffn(x_norm)
        x = x + self.dropout2(x_output)
        
        return x, new_cache