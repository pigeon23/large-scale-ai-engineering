from dataset import ParquetDataset, CollatorForCLM, IterableParquetDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")

# Create dataset instance
dataset_path = "/capstor/store/cscs/ethz/large-sc-2/datasets/train_data.parquet"
sequence_length = 4096
batch_size = 32

# Create dataset (only requesting 1 sample)
dataset = IterableParquetDataset(
    parquet_file=dataset_path,
    tokenizer=tokenizer,
    sequence_length=sequence_length,
    bos_token_id=-1
)

# Get a batch using a for loop 
i = 0
for item in dataset:
    continue
    print(item)
    batch_inputs, batch_labels = item
    # Print shapes
    print(f"Input shape: {batch_inputs.shape}")
    print(f"Labels shape: {batch_labels.shape}")

    # Count ignored tokens in the loss calculation
    ignored_count = (batch_labels == -100).sum().item()
    total_label_tokens = batch_labels.numel()
    print(f"Ignored tokens in loss: {ignored_count} out of {total_label_tokens} ({ignored_count/total_label_tokens*100:.2f}%)")

    # Only process the first batch
    break
