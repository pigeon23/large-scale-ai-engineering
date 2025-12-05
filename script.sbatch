#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --time=00:59:59
#SBATCH --job-name=lsai
#SBATCH --output=/iopsstor/scratch/cscs/yiswang/large-sc/project/logs/%x-%j.out
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --environment=/iopsstor/scratch/cscs/yiswang/large-sc/ngc_pt_jan.toml   # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

# Stop the script if a command fails or if an undefined variable is used
set -eo pipefail

export NCCL_NET="AWS Libfabric"

# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install datasets

echo "START TIME: $(date)"

ASSIGNMENT_DIR="/iopsstor/scratch/cscs/$USER/large-sc/project"

# The sbatch script is executed by only one node.
echo "[sbatch-master] running on $(hostname)"

echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"

echo "[sbatch-master] define some env vars that will be passed to the compute nodes"

# The defined environment vars will be shared with the other compute nodes.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)  
export MASTER_PORT=12345   # Choose an unused port
export FOOBAR=666
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

echo "[sbatch-master] execute command on compute nodes"

# The command that will run on each node
CMD="
# print current environment variables
echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

# run the script
numactl --membind=0-3 torchrun \
    --nnodes="${SLURM_NNODES}" \
    --node_rank=\$SLURM_NODEID \
    --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${ASSIGNMENT_DIR}"/train.py \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --lr-warmup-steps 100 \
    --training-steps 1000 \
    --sequence-length 2048 \
    --compile \
    --fused-optimizer \
    --tp-size 1 \
    --checkpoint-frequency 200 \
#     --checkpoint-save-path /iopsstor/scratch/cscs/yiswang/large-sc/project/ckpt \
#     --checkpoint-load-path /iopsstor/scratch/cscs/yiswang/large-sc/project/ckpt
"

# Submits the CMD to all the processes on all the nodes.
srun bash -c "$CMD"

echo "END TIME: $(date)"
echo "[sbatch-master] task finished"