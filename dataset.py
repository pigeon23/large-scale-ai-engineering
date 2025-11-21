import pyarrow.parquet as pq 
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer 

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
        bos_token_id: int = -1,
    ):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.bos_token_id = bos_token_id
        self.current_index = 0
        self.token_buffer = []
        
    def __len__(self):
        return self.real_length
    
    def __iter__(self):
        # Reset buffer and index when starting a new iteration
        self.token_buffer = []
        self.current_index = 0
        return self
    
    def __next__(self):
        # Keep filling a buffer until we have enough tokens for a new sample.
        # Mask the loss for each token following the BoS token using -100 index.
        
        # Add your implementation here
        if self.current_index == self.real_length:
          raise StopIteration

        while True:
          sample_str = str(self.parquet_ds["text"][self.current_index])
          ids = self.tokenizer.encode(sample_str)
          self.current_index += 1
          self.token_buffer.extend(ids)
          if not self.current_index == self.real_length:
            self.token_buffer.append(self.bos_token_id)
          if len(self.token_buffer) >= self.sequence_length or self.current_index == self.real_length:
            break

        inputs = self.token_buffer[: self.sequence_length]
        self.token_buffer = self.token_buffer[self.sequence_length:]
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

