from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch
from torch import nn
import re
import numpy as np
import pandas as pd

@dataclass
class SpecialTokens:
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"    # beginning of sentence
    eos_token: str = "<EOS>"    # end of sentence
    bop_token: str = "<BOP>"    # beginning of paragraph
    eop_token: str = "<EOP>"    # end of paragraph

@dataclass
class ModelConfig:
    lowercase: bool = True
    remove_punctuation: bool = False
    dropout_prob: float = 0.3

    # regex dùng để tách token
    split_sentence_pattern = re.compile(
        r'(?<=[.!?])\s+',
        re.UNICODE
    )

    # split token tiếng Việt cơ bản
    split_token_pattern = re.compile(
        r"""
        \d+(?:[.,:/-]\d+)*                 # numbers, dates, times
        | [^\W\d_]+(?:[-'][^\W\d_]+)*     # words
        | [^\w\s]                         # punctuation / symbols
        """,
        re.VERBOSE | re.UNICODE
    )

    # punctuation unicode-aware
    punctuation_pattern = r'[^\w\s]'

    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)

    vocab_size: int = 30000
    vocab_min_count: int = 5        # deprecated

    embed_dim: int = 512

    max_target_length: int = 1000



# -------------------------
# Pre-Process Pipeline: Vocabulary and Tokenizer Interfaces
#
# Input:    Corpura (raw text - paragraphs)
# Output:   Tokenized corpora (list of sentences, each sentence is a list of tokens)
#           Vocabulary (token to ID mapping, ID to token list, token frequency)
# -------------------------

class Tokenizer(ABC):
    """
    Interface for tokenization and normalization of text.

    Input:  Paragraph (str)
    Output: List of tokens.
    """

    @abstractmethod
    def tokenize(self, paragraph: str) -> list[str]:
        """Tokenizes a paragraph into a list of tokens"""
        pass

    def tokenize_batch(self, paragraphs: list[str]) -> list[list[str]]:
        """Tokenizes a batch of paragraphs."""
        tokenized_paragraphs = []
        
        for paragraph in paragraphs:
            tokenized_paragraphs.append(self.tokenize(paragraph))
            
        return tokenized_paragraphs

class Vocabulary(ABC):
    """
    Interface for building a vocabulary from tokenized corpora.

    Input:  Tokenized corpora (list of sentences, each sentence is a list of tokens)
    Output: Vocabulary (token to ID mapping, ID to token list, token frequency)
    """

    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def pad_idx(self) -> int:
        pass

    @abstractmethod
    def build_vocabulary(self, tokenized_corpora: list[list[str]], min_count: int, max_vocab_size: int) -> list[str]:
        """
        Builds the vocabulary from the tokenized corpora.
        Input: corpora (in tokenized form)
        Output: list of unique tokens (vocabulary) with their index as ID.
        """
        pass

    @abstractmethod
    def token_distribution(self) -> np.array:
        pass

    @abstractmethod
    def to_id(self, token: str) -> int:
        """Returns the ID of a token."""
        pass

    @abstractmethod
    def to_token(self, id: int) -> str:
        """Returns the token corresponding to an ID."""
        pass

    def token_to_id_sample(self, paragraph: list[str]) -> list[int]:
        return [self.to_id(token) for token in paragraph]

    def id_to_token_sample(self, paragraph: list[int]) -> list[str]:
        return [self.to_token(id) for id in paragraph]

    def token_to_id_batch(self, paragraphs: list[list[str]]) -> list[list[int]]:
        """Converts a batch of tokens to their corresponding IDs."""
        return [self.token_to_id_sample(p) for p in paragraphs]
    
    def id_to_token_batch(self, paragraphs: list[list[str]]) -> list[list[str]]:
        """Converts a batch of IDs to their corresponding tokens."""
        return [self.id_to_token_sample(p) for p in paragraphs]
    


# -------------------------
# Word Embeddings Interface
# 
# Input:    Corpura (raw text - paragraphs)
#           Vocabulary (token to id mapping)
# Output:   Word Embeddings (token to vector mapping, ID to vector list)
# ------------------------
    

class WordEmbedding(ABC, nn.Module):
    """
    Interface for creating an embedding layer that maps tokens to their corresponding vector representations.

    Input:  Vocabulary (token to ID mapping)
    Output: Word Embeddings (token to vector mapping, ID to vector list)
    """

    @abstractmethod
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns the embedding vectors for a batch of token IDs.
        
        Input: token_ids [batch_size, seq_length]
        Output: embeddings [batch_size, seq_length, embedding_dim]
        """
        pass

    @abstractmethod
    def get_embedding_matrix(self) -> torch.Tensor:
        """
        Extracts the embedding matrix E for other use.
        """
        pass

    from abc import ABC, abstractmethod
from typing import Dict, Sequence, Tuple


class evaluate(ABC):
    """
    Interface cho các lớp đánh giá mô hình tóm tắt văn bản.

    Class triển khai cụ thể cần tính điểm giữa:
    - reference: bản tóm tắt đúng, ví dụ cột summary
    - prediction: bản tóm tắt mô hình sinh ra, ví dụ cột pred_summary

    Các độ đo mặc định:
    - ROUGE-1: unigram overlap
    - ROUGE-2: bigram overlap
    - ROUGE-L: longest common subsequence
    - ROUGE-S: skip-bigram overlap
    """

    rouge_types: Tuple[str, ...] = (
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeS",
    )

    max_skip: int = 4

    @abstractmethod
    def score_one(
        self,
        reference: str,
        prediction: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Tính điểm ROUGE cho một cặp reference - prediction.

        Returns:
            {
                "rouge1": {"precision": ..., "recall": ..., "f1": ...},
                "rouge2": {"precision": ..., "recall": ..., "f1": ...},
                "rougeL": {"precision": ..., "recall": ..., "f1": ...},
                "rougeS": {"precision": ..., "recall": ..., "f1": ...}
            }
        """
        raise NotImplementedError

    @abstractmethod
    def score_batch(
        self,
        references: Sequence[str],
        predictions: Sequence[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Tính điểm ROUGE trung bình cho nhiều cặp reference - prediction.

        Args:
            references: Danh sách bản tóm tắt đúng.
            predictions: Danh sách bản tóm tắt mô hình sinh ra.

        Returns:
            Điểm trung bình ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-S.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset trạng thái nếu evaluator có lưu cache hoặc thống kê tạm.
        Với ROUGE đơn giản có thể để pass trong class triển khai.
        """
        raise NotImplementedError