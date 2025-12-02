import pyarrow.parquet as pq 
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer 

import math

class ParquetDataset(Dataset):
  def __init__(self, parquet_file: str, tokenizer: str, sequence_length: int, training_samples: int):
    self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
    self.real_length = len(self.parquet_ds)
    self.tokenizer = tokenizer
    self.sequence_length = sequence_length
    self.training_samples = training_samples

  def __len__(self):
    return self.training_samples
  
  def __getitem__(self, idx: int):
    sample_str = str(self.parquet_ds["text"][idx % self.real_length])
    return self.tokenizer.encode_plus(sample_str,
                                      max_length=self.sequence_length + 1,
                                      padding='max_length',
                                      truncation=True,
                                      padding_side="right")

class IterableParquetDataset(IterableDataset):
    def __init__(
        self,
        parquet_file: str,
        tokenizer,
        sequence_length: int,
        dp_rank: int,
        dp_size: int,
        bos_token_id: int = 1
    ):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.bos_token_id = bos_token_id
        self.token_buffer = []

        self.dp_rank = dp_rank
        self.dp_size = dp_size
        
    def __iter__(self):
        self.token_buffer = []
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            self.dp_rank = self.dp_rank * num_workers + worker_id
            self.dp_size *= num_workers
        
        dp_group_length = math.ceil(self.real_length / self.dp_size)
        
        self.start_idx = self.dp_rank 
        self.end_idx = self.start_idx + (dp_group_length - 1) * self.dp_size 
        self.end_idx = min(self.end_idx, self.real_length)
        
        self.current_index = self.start_idx
        
        return self
        
    def __next__(self):
        while True:
            if self.current_index > self.end_idx:
                self.current_index = self.start_idx
            
            sample_str = str(self.parquet_ds["text"][self.current_index])
            self.current_index += self.dp_size
            
            if self.token_buffer and self.token_buffer[-1] != self.bos_token_id:
                self.token_buffer.append(self.bos_token_id)
                
            tokens = self.tokenizer.encode(sample_str)
            self.token_buffer.extend(tokens)
            
            if len(self.token_buffer) >= self.sequence_length + 1:
                break
        
        inputs = self.token_buffer[:self.sequence_length+1]
        
        self.token_buffer = self.token_buffer[self.sequence_length+1:]
        inputs_id = torch.LongTensor(inputs)
        inputs = inputs_id[:-1].clone()
        labels = inputs_id[1:]
        
        labels[inputs == self.bos_token_id] = -100
        
        return inputs, labels

from dataclasses import dataclass
from typing import List, Dict
import torch

@dataclass
class CollatorForCLM:
  sequence_length: int
  pad_token_id: int
  def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.LongTensor([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s+1)

    inputs = input_ids[:, :-1].clone()
    labels = input_ids[:, 1:]

    # For padding tokens, mask the loss
    labels[labels == self.pad_token_id] = -100

    assert inputs.shape[1] == labels.shape[1] == self.sequence_length
    assert inputs.shape == labels.shape

    return inputs, labels

