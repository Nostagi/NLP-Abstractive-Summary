from .interfaces import ModelConfig

import pandas as pd
import os
from typing import List, Iterator
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder
from tokenizers.normalizers import Lowercase
from tokenizers.normalizers import Sequence as NormalizerSequence
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
    

class BPETokenizer:
    """
    Class quản lý Tokenizer sử dụng thuật toán Byte-Pair Encoding (BPE).
    Được tối ưu hóa bằng lõi Rust của thư viện Hugging Face 'tokenizers'.
    """

    def __init__(self, config: ModelConfig):
        """
        Khởi tạo kiến trúc cho BPETokenizer.
        
        Args:
            config (ModelConfig): Đối tượng chứa các tham số cấu hình cho tokenizer.
        """
        self.vocab_size = config.vocab_size
        self.special_tokens = config.special_tokens

        # 1. Khởi tạo mô hình BPE trống (chưa có từ điển)
        self.tokenizer = Tokenizer(BPE(unk_token=self.special_tokens.unk_token))

        # 2. Cài đặt Normalizer: Chuẩn hóa văn bản (ví dụ: đưa về chữ thường)
        # Sequence cho phép bạn nối nhiều bộ chuẩn hóa lại với nhau
        self.tokenizer.normalizer = NormalizerSequence([Lowercase()]) if config.lowercase else None

        # 3. Cài đặt Pre-tokenizer kết hợp Whitespace và Punctuation
        self.tokenizer.pre_tokenizer = PreTokenizerSequence([Whitespace(), Punctuation()])

        # 4. Cài đặt Decoder: Giúp ghép các Subword lại thành văn bản hoàn chỉnh khi decode
        self.tokenizer.decoder = BPEDecoder(suffix="</w>")

    def train_from_iterator(self, iterator: Iterator[str]) -> None:
        """
        Huấn luyện Tokenizer (xây dựng từ điển và các luật merge) từ một tập dữ liệu văn bản.
        
        Args:
            iterator (Iterator[str]): Generator hoặc List chứa các câu văn bản raw.
        """
        print(f"[Info] Đang huấn luyện BPE Tokenizer với vocab_size={self.vocab_size}...")
        
        # Khởi tạo Trainer với kích thước từ điển và danh sách special tokens
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens.as_list(),
            end_of_word_suffix="</w>",
            show_progress=True
        )

        # Bắt đầu train
        self.tokenizer.train_from_iterator(iterator, trainer=trainer)
        print(f"[Info] Hoàn tất huấn luyện! Kích thước từ điển thực tế: {self.tokenizer.get_vocab_size()}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Chuyển đổi một chuỗi văn bản thành danh sách các token IDs.
        Ủy thác hoàn toàn việc kẹp special token cho lõi Rust.
        """
        # Thư viện gốc đã hỗ trợ tham số add_special_tokens
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode hàng loạt câu cùng lúc (Tận dụng tối đa đa luồng của Rust).
        """
        outputs = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [output.ids for output in outputs]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Dịch ngược từ danh sách IDs về lại văn bản con người đọc được.
        """
        # Hàm này hiện tại của bạn đã rất ngắn gọn và chuẩn xác rồi
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def token_to_id(self, token: str) -> int:
        """Lấy ID của một token string."""
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> str:
        """Lấy string token tương ứng với ID."""
        return self.tokenizer.id_to_token(id)

    def save(self, base_dir: str) :
    
        save_folder = os.path.join(base_dir, "tokenizer")
        os.makedirs(save_folder, exist_ok=True)

        # 1. Trích xuất và lưu riêng Vocab ra file CSV
        csv_path = os.path.join(save_folder, "vocab.csv")
        vocab_dict = self.tokenizer.get_vocab() # Trả về dictionary {token: id}
        df_vocab = pd.DataFrame(list(vocab_dict.items()), columns=['Token', 'ID']).set_index('ID').sort_index()

        # 2. Lưu toàn bộ pipeline của Tokenizer (JSON chuẩn của Hugging Face)
        json_path = os.path.join(save_folder, "tokenizer.json")
        self.tokenizer.save(json_path)
        df_vocab.to_csv(csv_path, index=False)
        
        print(f"[Info] Đã lưu cấu trúc Tokenizer và Vocab (CSV) tại: {save_folder}")
        return df_vocab

    @classmethod
    def load(cls, save_folder: str, config:ModelConfig=None) -> 'BPETokenizer':
        """
        Tải lại Tokenizer từ folder đã lưu (chỉ cần đọc file JSON).
        """
        json_path = os.path.join(save_folder, "tokenizer.json")

        if config is None:
            config = ModelConfig()
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"[ERROR] Không tìm thấy file tokenizer.json tại: {save_folder}")
            
        instance = cls(config)
        instance.tokenizer = Tokenizer.from_file(json_path)
        print(f"[Info] Đã tải thành công Tokenizer từ: {save_folder}")
        
        return instance