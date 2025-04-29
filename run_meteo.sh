#!/bin/bash -l
#SBATCH --job-name=MeteoGraphSAGE_v2
#SBATCH --partition=tfg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --mem=29G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Carregar Conda en non-interactive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfg_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONUNBUFFERED=1

mkdir -p $SLURM_SUBMIT_DIR/logs

echo "=== GPUs assignades ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_LL_THRESHOLD=0

torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  MeteoGraphSAGE_v2.py \
    --data_dir /fhome/nfarres/DADES_METEO_PC_TO_DATA_v4 \
    --group_by day \
    --history_length 48 \
    --stride 24 \
    --target_variable Temp \
    --epochs 100 \
    --hidden_dim 128 \
    --graph_layers 3 \
    --station_embedding_dim 16 \
    --aggregator gat \
    --temporal_model gru \
    --num_attention_heads 4 \
    --learning_rate 5e-4 \
    --out_all_vars \
    --predict_station ESCAT0800000008600A \
    --output_netcdf
