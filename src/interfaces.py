from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass, field
from typing import Dict, List
import os
import json
import torch
from torch import nn

@dataclass
class SpecialTokens:
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    eos_token: str = "</s>"    # end of sentence (sample)

    def as_list(self) -> List[str]:
        return list(astuple(self))

@dataclass
class ModelConfig:
    lowercase: bool = True
    dropout_prob: float = 0.1

    vocab_size: int = 30000
    embed_dim: int = 512
    max_target_length: int = 1000

    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)

    evaluation_config: Dict = field(default_factory=lambda: {
            "rouge": {
                "rouge_types": ["rouge1", "rouge2", "rouge3", "rougeLsum"],
                "use_stemmer": True
            },
            "bertscore": {
                # Vì tóm tắt tiếng Việt, ta nên dùng PhoBERT để đo khoảng cách ngữ nghĩa
                "model_type": "vinai/phobert-base", 
                "num_layers": 9 # Tham số sâu của BERTScore (optional)
            }
        }
        )




# -------------------------
# Network (Model) Base Class
#
# Cung cấp sơ bộ các hàm save/load model.
#  
# Input:    Indices (token IDs) của câu tóm tắt được tạo ra bởi Tokenizer
# Output:   Indices (token IDs) của câu tóm tắt được tạo ra bởi mô hình
# ------------------------

class Network(ABC, nn.Module):
    
    @abstractmethod
    def get_config(self) -> dict:
        """
        Hàm trừu tượng: Các model kế thừa phải trả về một dictionary chứa 
        các tham số dùng để khởi tạo model (trong hàm __init__).
        """
        pass

    def save_state(self, save_folder: str, optimizer=None, scheduler=None) -> str:
        """
        Lưu trạng thái và cấu hình mạng vào một folder con sinh theo thời gian thực.
        
        Args:
            save_folder: Đường dẫn tới thư mục được tạo.
            optimizer: Đối tượng optimizer từ PyTorch (tùy chọn).
            scheduler: Đối tượng learning rate scheduler từ PyTorch (tùy chọn).
        Returns:
            
        """

        os.makedirs(save_folder, exist_ok=True)
        
        # 1. Lưu trọng số (state_dict)
        weights_path = os.path.join(save_folder, "model_state.pth")
        torch.save(self.state_dict(), weights_path)
        
        # 2. Lưu cấu trúc (config) dưới dạng file JSON
        config_path = os.path.join(save_folder, "model_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_config(), f, indent=4)

        # 3. Lưu trạng thái Optimizer (nếu có)
        if optimizer is not None:
            optimizer_path = os.path.join(save_folder, "optimizer_state.pth")
            torch.save(optimizer.state_dict(), optimizer_path)
            
        # 4. Lưu trạng thái Scheduler (nếu có)
        if scheduler is not None:
            scheduler_path = os.path.join(save_folder, "scheduler_state.pth")
            torch.save(scheduler.state_dict(), scheduler_path)
            
        print(f"[Info] Đã lưu model thành công tại: {save_folder}")
        return save_folder

    @classmethod
    def load_state(cls, save_folder: str, device: str = 'cpu'):
        """
        Đọc cấu hình, tái tạo kiến trúc và nạp trọng số từ folder đã lưu.
        
        Args:
            save_folder: Đường dẫn tới thư mục lưu trữ (thư mục chứa pth và json).
            device: Nơi chứa model ('cpu' hoặc 'cuda').
        Returns:
            Mô hình đã được nạp sẵn kiến trúc và trọng số.
        """
        weights_path = os.path.join(save_folder, "model_state.pth")
        config_path = os.path.join(save_folder, "model_config.json")
        optimizer_path = os.path.join(save_folder, "optimizer_state.pth")
        scheduler_path = os.path.join(save_folder, "scheduler_state.pth")
        
        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            raise FileNotFoundError(f"[ERROR] Không tìm thấy đủ file pth và json tại: {save_folder}")
            
        # 1. Đọc config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # 2. Khởi tạo lại object từ class gọi hàm bằng unpacked kwargs (**config)
        # Tương đương với: model = Transformer(vocab_size=..., d_model=...)
        model = cls(**config)
        
        # 3. Nạp trọng số
        state_dict = torch.load(weights_path, map_location=torch.device(device), weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        
        optimizer_state = None
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path, map_location=torch.device(device))
        
        scheduler_state = None
        if os.path.exists(scheduler_path):
            scheduler_state = torch.load(scheduler_path, map_location=torch.device(device))
            
        print(f"[Info] Đã tái tạo và tải trạng thái mô hình thành công từ: {save_folder}")
        if optimizer_state:
            print("       - Tìm thấy trạng thái Optimizer.")
        if scheduler_state:
            print("       - Tìm thấy trạng thái Scheduler.")
            
        return model, optimizer_state, scheduler_state

    @abstractmethod
    def generate(self, source_ids: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 150, device: torch.device = None) -> torch.Tensor:
        """
        Hàm trừu tượng: Các model kế thừa phải cài đặt hàm này để sinh ra câu tóm tắt từ input token IDs.

        Args:
            source_ids: Tensor chứa token IDs của câu gốc (shape: [batch_size, seq_len]).
            bos_id: ID của token <BOS>.
            eos_id: ID của token <EOS>.
            max_len: Độ dài tối đa của câu tóm tắt được sinh ra.
            device: Thiết bị để thực hiện tính toán (CPU hoặc GPU).
        """
        pass

