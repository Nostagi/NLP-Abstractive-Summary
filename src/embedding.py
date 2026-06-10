import torch
import torch.nn as nn
import math

    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, dropout:float = 0.1, max_seq_len:int = 5000, 
                 bypass:bool = False):
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
        self.bypass = bypass

    def set_bypass(self, bypass:bool) :
        self.bypass = bypass

    def forward(self, seq_v: torch.Tensor, step: int = 0) -> torch.Tensor:
        if self.bypass :
            return seq_v

        x = seq_v * math.sqrt(self.d_model) + self.pe[:, step : step + seq_v.size(1)]

        return self.dropout(x)
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2000, base: int = 10000):
        """
        Params:
            head_dim: Số chiều của MỘT head (tức là d_k = d_model // num_heads).
            max_seq_len: Độ dài câu tối đa.
        """
        super().__init__()
        self.head_dim = head_dim

        # 1. Tính toán các chu kỳ tần số (theta_i)
        # Công thức: theta_i = 10000^(-2(i-1)/d)
        # Shape của inv_freq: [dim // 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

        # 2. Tạo mảng các vị trí m = [0, 1, 2, ..., max_seq_len - 1]
        # Shape của t: [max_seq_len]
        t = torch.arange(max_seq_len, dtype=torch.float32)

        # 3. Tính m * theta_i bằng outer product (einsum)
        # Shape của freqs: [max_seq_len, dim // 2]
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # 4. Nhân đôi freqs để khớp với chiều dim. 
        # Cấu trúc: nửa đầu [0:dim/2] và nửa sau [dim/2:dim] giống nhau.
        # Kiểu ghép nối (cat) này tuân theo implementation chuẩn của HuggingFace (LLaMA).
        # Shape của emb: [max_seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        # 5. Lấy Cosine và Sine, thêm các chiều giả (unsqueeze) để tiện cho việc broadcasting sau này
        # Mở rộng chiều thành: [batch_size=1, num_heads=1, seq_len, dim]
        cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)
        sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)

        # register_buffer giúp lưu các tensor này cùng với model state_dict 
        # nhưng không tính gradient (không phải là tham số học được).
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hàm xoay vector q và k. 
        Biến đổi x = [x1, x2, ..., x_d/2, x_d/2+1, ..., x_d] 
        Thành: [-x_d/2+1, ..., -x_d, x1, x2, ..., x_d/2]
        """
        # Chia đôi x theo chiều cuối cùng
        # Shape của x1, x2: [batch_size, num_heads, seq_len, dim // 2]
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        
        # Ghép lại với nửa thứ 2 bị đảo dấu
        # Shape đầu ra: [batch_size, num_heads, seq_len, dim]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                step: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Áp dụng xoay RoPE lên Query và Key.
        Input:
            q: Tensor shape [batch_size, num_heads, seq_len, dim]
            k: Tensor shape [batch_size, num_heads, seq_len, dim]
        Output:
            q_out, k_out: Cùng shape với q và k.
        """
        seq_len = q.size(-2) # Lấy seq_len hiện tại của q
        
        # Lấy cos và sin từ cache với đúng độ dài câu
        # Shape của cos, sin: [1, 1, seq_len, dim]
        cos = self.cos_cached[:, :, step : step + seq_len, ...]
        sin = self.sin_cached[:, :, step : step + seq_len, ...]
        
        # Áp dụng công thức RoPE: q * cos + rotate_half(q) * sin
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_rot, k_rot
    
