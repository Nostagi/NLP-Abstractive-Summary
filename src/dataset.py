from typing import Callable, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SummarizationDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, encode_fn: Callable[[List[str]], List[List[int]]],
                 source='article', target='summary',):
        
        source = dataset[source].astype(str).tolist()
        target = dataset[target].astype(str).tolist()

        self.source_ids = encode_fn(source)
        self.target_ids = encode_fn(target)

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