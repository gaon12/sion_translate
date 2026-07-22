#!/usr/bin/env bash
#SBATCH --job-name=kjx-data-fit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export OMP_NUM_THREADS=8
export KJX_CONFIG="${KJX_CONFIG:-configs/kjx_data_fit.yaml}"

srun --label bash -c '
torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc-per-node=8 \
  --node-rank="$SLURM_NODEID" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  -m kjx.cli.train --config "$KJX_CONFIG"
'
