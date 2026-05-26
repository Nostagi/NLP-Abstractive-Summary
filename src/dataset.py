from .interfaces import Vocabulary, Tokenizer

from typing import List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SGNSDataset(Dataset):

    def __init__(self, vocab:Vocabulary, 
                 token_indices: List[List[int]],
                 window_size: int = 5, negative_sampling: int = 5,
    ):
        self.token_indices = token_indices
        self.vocab_size = len(vocab)

        self.window_size = window_size
        self.negative_size = negative_sampling

        # negative sampling distribution
        freq = vocab.token_distribution()

        distribution = np.power(freq, 0.75)
        self.distribution = distribution / sum(distribution)

        # random generator
        self.rng = np.random.default_rng()      

    def __len__(self):

        return len(self.token_indices)

    def __getitem__(self, idx):

        doc = self.token_indices[idx]
        doc_length = len(doc)

        center_samples = []
        positive_samples = []
        negative_samples = []

        for i, center in enumerate(doc):
            left = max(0, i - self.window_size)
            right = min(doc_length, i + self.window_size + 1)

            positive = doc[left:i] + doc[i+1:right]

            if len(positive) == 0:
                continue

            for context in positive:
                negative = self.rng.choice(
                        a    = self.vocab_size,
                        size = self.negative_size,
                        p    = self.distribution
                    ).tolist()
                
                center_samples.append(center)
                positive_samples.append(context)
                negative_samples.append(negative)

        return {
            "center": torch.tensor(center_samples, dtype=torch.long),       # (N, )
            "positive": torch.tensor(positive_samples, dtype=torch.long),   # (N, )
            "negative": torch.tensor(negative_samples, dtype=torch.long),   # (N, K)
        }   

    @staticmethod
    def collate_fn(batch):

        centers = []
        positives = []
        negatives = []

        for item in batch:
            centers.append(item["center"])       # (N,)
            positives.append(item["positive"])   # (N, )
            negatives.append(item["negative"])   # (N_i, K)

        centers = torch.cat(centers, dim=0)     # (N,)
        contexts = torch.cat(positives, dim=0)   # (N, )
        negatives = torch.cat(negatives, dim=0) # (N, K)

        return {
            "center": centers,
            "context": contexts,
            "negative": negatives
        }
    


class SummarizationDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, vocab:Vocabulary, tokenizer:Tokenizer,
                 source='article', target='summary',):
        
        source = dataset[source].astype(str).tolist()
        target = dataset[target].astype(str).tolist()

        source = tokenizer.tokenize_batch(source)
        target = tokenizer.tokenize_batch(target)

        self.source_ids = vocab.token_to_id_batch(source)
        self.target_ids = vocab.token_to_id_batch(target)

        self.max_source_len = max([len(seq) for seq in self.source_ids])
        self.max_target_len = max([len(seq) for seq in self.target_ids])


    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.source_ids[idx], dtype=torch.long),
            torch.tensor(self.target_ids[idx], dtype=torch.long),
        )
        
    
class TransformerCollate:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):

        source_batch, target_batch = zip(*batch)
        
        # pad_sequence chuẩn hóa shape: [batch_size, max_seq_len_in_batch]
        source_padded = pad_sequence(source_batch, batch_first=True, padding_value=self.pad_id)
        target_padded = pad_sequence(target_batch, batch_first=True, padding_value=self.pad_id)
        
        return source_padded, target_padded