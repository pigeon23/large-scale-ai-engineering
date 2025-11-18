import os
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed._composable.fsdp import fully_shard
from transformers import AutoTokenizer
from torch.distributed.tensor.parallel import loss_parallel

from dataset import CollatorForCLM, ParquetDataset
from model import Transformer, TransformerModelArgs, apply_tensor_parallel
# Removed init_distributed from utils as we handle it explicitly with DeviceMesh now
from utils import build_lr_scheduler, clip_grad_norm_, get_args, get_num_params, get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype

def train(args):
  logger.info(f"Experiment args: {args}")
  
  # --- Init Distributed Environment & Device Mesh ---
  # 1. Initialize the standard process group first (required to get world size/rank info)
  dist.init_process_group(backend="nccl")
  
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  local_rank = int(os.environ["LOCAL_RANK"])
  
  device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
  torch.cuda.set_device(device)
  
  # 2. Initialize DeviceMesh with Tensor Parallel dimension
  # We check for args.tp_size, defaulting to 1 if not specified
  tp_size = getattr(args, "tp_size", 1)
  dp_size = world_size // tp_size
  
  if world_size % tp_size != 0:
      raise ValueError(f"World size {world_size} must be divisible by TP size {tp_size}")
  
  # Create a 2D mesh: (Data Parallel, Tensor Parallel)
  # "data" dim: used for DDP gradient synchronization
  # "tensor" dim: used for intra-layer tensor parallelism (if implemented in model)
  mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("data", "tensor"))
  
  # Get the coordinate of the current rank in the mesh (dp_rank, tp_rank)
  # Note: if tp_size=1, the structure is effectively 1D but coordinates are still consistent
  dp_rank = mesh.get_coordinate()[0]
  
  model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]

  # Set up DataLoader
  logger.info("Setting up DataLoaders...")
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
  
  # Total tokens logic should now depend on dp_size (number of independent data streams)
  # rather than world_size (total GPUs)
  train_ds = ParquetDataset(args.dataset, tokenizer, args.sequence_length, dp_size*args.batch_size*args.training_steps)
  train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)
  
  # DistributedSampler needs to know the DP rank and size.
  # Ranks within the same TP group (same DP rank) will receive the SAME data batch.
  train_dl = DataLoader(train_ds,
                        batch_size=args.batch_size,
                        sampler=DistributedSampler(train_ds, num_replicas=dp_size, rank=dp_rank, shuffle=True),
                        collate_fn=train_collator)
  

  # Set up Model
  logger.info("Setting up Model...")
  model_config = TransformerModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=4,
        n_kv_heads=4,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        vocab_size=tokenizer.vocab_size,
        seq_len=args.sequence_length,
    )
  
  with set_default_dtype(model_dtype):
    model = Transformer(model_config).to(device)
    if tp_size > 1:
      logger.info(f"Applying Tensor Parallelism with size {tp_size}...")
      model = apply_tensor_parallel(model, mesh['tensor'])
    # Pass the mesh's "data" process group to DDP
    # This ensures DDP syncs gradients only across the "data" dimension
    # (i.e., across DP ranks, but not across TP ranks)
    model = fully_shard(model, mesh=mesh['data'])

  
  # if args.compile:
  #   logger.info("Using `torch.compile`")
  #   model = torch.compile(model, fullgraph=True)
  
  model.train()

  # Build Optimizers & LR Scheduler
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer)
  lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)

  # Utils
  num_flop_per_token = get_num_flop_per_token(
      get_num_params(model, exclude_embedding=True),
      model_config,
  )

  ntokens_since_last_log = 0
  ntraining_tokens_since_last_log = 0
  time_last_log = time.perf_counter()

  logger.info("Starting training!")
  train_step = 0
  for input_ids, labels in train_dl:
    train_step += 1
    ntokens_since_last_log += args.batch_size * args.sequence_length * dp_size
    num_items_in_batch = labels.ne(-100).sum()
    ntraining_tokens_since_last_log += num_items_in_batch
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    logits = model(input_ids)
    
    with loss_parallel():
      loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1))
      loss.backward()
    
    # --- START: Loss Aggregation Logic for Logging ---
    reduced_loss = loss.clone().detach()
    average_loss = reduced_loss.item() 
    
    # print(f"[Rank {rank}] loss before all_reduce: {loss}")
    
    if world_size > 1:
      with torch.no_grad():
        if isinstance(loss, torch.distributed.tensor.DTensor):
            loss_log = loss.full_tensor()
        else:
            loss_log = loss

        # Reduce across Data Parallel dimension only
        dist.all_reduce(loss_log, op=dist.ReduceOp.SUM, group=mesh.get_group("data"))
        average_loss = loss_log / mesh["data"].size()
        average_loss = average_loss.item()

    del logits
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    lr_scheduler.step()
    
    # Logging
    if (train_step == 1 or train_step % args.logging_frequency == 0) and rank == 0:
      time_delta = time.perf_counter() - time_last_log
      tps = ntokens_since_last_log / time_delta 
      mfu = 100 * num_flop_per_token * tps / 989e12
      tflops = num_flop_per_token * tps / 1e12
      training_tps = ntraining_tokens_since_last_log / time_delta
      logger.info(f"Step: {train_step} | Loss (Avg): {average_loss:.2f} | Tokens per second: {tps:.2f} | Training tokens per second (%): {100*training_tps/tps:.2f} | MFU (%): {mfu:.2f} | TFLOPs: {tflops:.2f}")
      ntokens_since_last_log = 0
      ntraining_tokens_since_last_log = 0
      time_last_log = time.perf_counter()
    
  logger.info("Training completed")
  torch.distributed.destroy_process_group()

if __name__ == "__main__":
  init_logger()
  args = get_args()
  train(args)