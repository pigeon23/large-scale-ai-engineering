import os
import time
import numpy as np

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import AutoTokenizer
from torch.distributed.tensor.parallel import loss_parallel
import torch.distributed.checkpoint as dcp

from dataset import CollatorForCLM, ParquetDataset, IterableParquetDataset
from model import Transformer, TransformerModelArgs, apply_tensor_parallel
# Removed init_distributed from utils as we handle it explicitly with DeviceMesh now
from utils import init_weights, build_lr_scheduler, clip_grad_norm_, get_args, get_num_params, get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype
from checkpoint_manager import CheckpointManager

def train_iteration(input_ids, labels, model, optimizer, lr_scheduler, device, mesh, world_size):
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    
    # logger.info(f"[Rank {device}] Memory allocation: {round(torch.cuda.memory_allocated(device) / 1e9, 3)} GB; Max allocation: {round(torch.cuda.max_memory_allocated(device) / 1e9, 3)} GB")
    
    logits = model(input_ids)
    
    with loss_parallel():
      loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1))
      loss.backward()
    
    # --- START: Loss Aggregation Logic for Logging ---
    reduced_loss = loss.clone().detach()
    average_loss = reduced_loss.item() 
    
    # print(f"[Rank {rank}] loss before all_reduce: {loss}")
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # logger.info(f"[Rank {device}] Memory allocation: {round(torch.cuda.memory_allocated(device) / 1e9, 3)} GB; Max allocation: {round(torch.cuda.max_memory_allocated(device) / 1e9, 3)} GB")
    
    optimizer.step()
    lr_scheduler.step()
    
    return loss

def train(args):
  logger.info(f"Experiment args: {args}")
  
  # --- Init Distributed Environment & Device Mesh ---
  # 1. Initialize the standard process group first (required to get world size/rank info)
  local_rank = int(os.environ["LOCAL_RANK"])
  print(f"Setting device to local rank: {local_rank}")
  torch.cuda.set_device(local_rank)
  device = torch.device(f"cuda:{local_rank}")
  
  dist.init_process_group(backend="nccl")
  
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  
  # 2. Initialize DeviceMesh with Tensor Parallel dimension
  # We check for args.tp_size, defaulting to 1 if not specified
  tp_size = getattr(args, "tp_size", 1)
  dp_size = world_size // tp_size
  
  if world_size % tp_size != 0:
      raise ValueError(f"World size {world_size} must be divisible by TP size {tp_size}")
  
  # Create a 2D mesh: (Data Parallel, Tensor Parallel)
  # "data" dim: used for DP gradient synchronization
  # "tensor" dim: used for intra-layer tensor parallelism (if implemented in model)
  mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("data", "tensor"))
  
  # Get the coordinate of the current rank in the mesh (dp_rank, tp_rank)
  # Note: if tp_size=1, the structure is effectively 1D but coordinates are still consistent
  dp_rank = mesh.get_coordinate()[0]
  tp_rank = mesh.get_coordinate()[1]
  
  print(f"[Rank {rank}] World Size: {world_size}, DP: {dp_rank} / {dp_size}, TP: {tp_rank} / {tp_size}")
  
  model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]

  # Set up DataLoader
  logger.info("Setting up DataLoaders...")
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
  
  # TODO: each sample only contain one text with many paddings, but iterable dataset doesn't support DistributedSampler
  # train_ds = ParquetDataset(args.dataset, tokenizer, args.sequence_length, dp_size*args.batch_size*args.training_steps)
  # train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)  
  
  # DistributedSampler needs to know the DP rank and size.
  # Ranks within the same TP group (same DP rank) will receive the SAME data batch.
  # train_dl = DataLoader(train_ds,
  #                       batch_size=args.batch_size,
  #                       pin_memory=True,
  #                       num_workers=4,
  #                       sampler=DistributedSampler(train_ds, num_replicas=dp_size, rank=dp_rank, shuffle=True),
  #                       collate_fn=train_collator)
  
  train_ds = IterableParquetDataset(args.dataset, tokenizer, args.sequence_length, dp_rank, dp_size)
  
  train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
  

  # Set up Model
  logger.info("Setting up Model...")
  model_config = TransformerModelArgs(
        dim=4096,
        n_layers=16,
        n_heads=16,
        n_kv_heads=8,
        multiple_of=1024,
        rope_theta=500000,
        vocab_size=tokenizer.vocab_size,
        seq_len=args.sequence_length,
    )
  
  with set_default_dtype(model_dtype):
    with torch.device('meta'):
      model = Transformer(model_config)
    if tp_size > 1:
      logger.info(f"Applying Tensor Parallelism with size {tp_size}...")
      apply_tensor_parallel(model, mesh['tensor'])
    
     # Some problem with Embedding layer and output layer when compiling related to tp
    if args.compile:
      logger.info("Compiling model...")
      for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(transformer_block, fullgraph=True)
        model.layers.register_module(layer_id, transformer_block)
    
    mixture = MixedPrecisionPolicy(torch.bfloat16, torch.float32)
    for layer in model.layers.values():
        fully_shard(layer, mesh=mesh['data'], reshard_after_forward=True, mp_policy=mixture)
    fully_shard(model, mesh=mesh['data'], reshard_after_forward=False, mp_policy=mixture)
    model = model.to_empty(device=device)
    
    rng = torch.Generator(device=device)
    rng.manual_seed(tp_rank)
    
    init_weights(model, rng)
  
  # Build Optimizers & LR Scheduler
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer)
  lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)
  
  checkpoint_manager = CheckpointManager(model, optimizer)
  if args.checkpoint_load_path is not None and os.path.isdir(args.checkpoint_load_path):
    logger.info(f"Load checkpoint from {args.checkpoint_load_path}")
    state_dict = { "app": checkpoint_manager}
    dcp.load(
        state_dict=state_dict,
        checkpoint_id=args.checkpoint_load_path,
    )
  torch.distributed.barrier()
    
  if rank == 0 and args.checkpoint_save_path and not os.path.exists(args.checkpoint_save_path):
    os.mkdir(args.checkpoint_save_path)

  # Utils
  num_flop_per_token = get_num_flop_per_token(
      get_num_params(model, exclude_embedding=True),
      model_config,
  )

  ntokens_since_last_log = 0
  ntraining_tokens_since_last_log = 0
  time_last_log = time.perf_counter()
  
  # logger.info(f"[Rank {device}] Memory allocation: {round(torch.cuda.memory_allocated(device) / 1e9, 3)} GB; Max allocation: {round(torch.cuda.max_memory_allocated(device) / 1e9, 3)} GB")

  logger.info("Starting training!")
  model.train()
  train_step = 0
  for input_ids, labels in train_dl:
    if train_step > args.training_steps:
      break
    
    train_step += 1
    ntokens_since_last_log += args.batch_size * args.sequence_length * dp_size
    num_items_in_batch = labels.ne(-100).sum()
    ntraining_tokens_since_last_log += num_items_in_batch
      
    loss_tensor = train_iteration(input_ids, labels, model, optimizer, lr_scheduler, device, mesh, world_size)
    
    # Logging
    if (train_step == 1 or train_step % args.logging_frequency == 0):
      with torch.no_grad():
        if isinstance(loss_tensor, torch.distributed.tensor.DTensor):
            loss_log = loss_tensor.full_tensor()
        else:
            loss_log = loss_tensor
            
        # Reduce across Data Parallel dimension only
        dist.all_reduce(loss_log, op=dist.ReduceOp.SUM, group=mesh.get_group("data"))
        average_loss = loss_log / mesh["data"].size()
        average_loss = average_loss.item()
      
      time_delta = time.perf_counter() - time_last_log
      tps = ntokens_since_last_log / time_delta 
      mfu = 100 * num_flop_per_token * tps / (989e12 * world_size)
      tflops = num_flop_per_token * tps / 1e12 / world_size
      training_tps = ntraining_tokens_since_last_log / time_delta
      memory = torch.cuda.memory.memory_reserved() / 2**30
      if rank == 0:
        logger.info(f"Step: {train_step} | Loss (Avg): {average_loss:.2f} | Reserved Memory {memory:.2f} GB  | Tokens per second: {tps:.2f} | Training tokens per second (%): {100*training_tps/tps:.2f} | MFU (%): {mfu:.2f} | TFLOP/s/GPU: {tflops:.2f}")
      ntokens_since_last_log = 0
      ntraining_tokens_since_last_log = 0
      time_last_log = time.perf_counter()
      
      
    if train_step % args.checkpoint_frequency == 0 and args.checkpoint_save_path is not None:
      logger.info(f"Step: {train_step} | Save checkpoint into {args.checkpoint_save_path}")
      state_dict = { "app": checkpoint_manager}
      dcp.save(state_dict, checkpoint_id=args.checkpoint_save_path)
      torch.distributed.barrier()
      
    
  logger.info("Training completed")
  torch.distributed.destroy_process_group()

if __name__ == "__main__":
  init_logger()
  args = get_args()
  train(args)