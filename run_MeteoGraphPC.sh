#!/bin/bash -l
#SBATCH --job-name=MeteoGraphPC_Train
#SBATCH --partition=tfg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --mem=29G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ───────────────────────────────────────────────────────────────────────────── #
#  Configuració de CUDA i SLURM                                                 #
# ───────────────────────────────────────────────────────────────────────────── #
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($SLURM_GPUS_ON_NODE-1)))

# ───────────────────────────────────────────────────────────────────────────── #
#  Entorn Python / Conda                                                       #
# ───────────────────────────────────────────────────────────────────────────── #
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfg_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p $SLURM_SUBMIT_DIR/logs

# ───────────────────────────────────────────────────────────────────────────── #
#  Paràmetres de l’entrenament                                                  #
# ───────────────────────────────────────────────────────────────────────────── #
SEQ_DIR="/fhome/nfarres/DADES_METEO_PC_generated_seqs_ws168_str6_hh168"
BATCH_SIZE=4
EPOCHS=20
LR=1e-3
HIDDEN_DIM=256
PATIENCE=15
MIN_DELTA=1e-4
DEVICE="cuda"
GRAD_CLIP=5.0
STD_EPS=1e-6
DL_NUM_WORKERS=6
MODEL="dyngcn"

# ───────────────────────────────────────────────────────────────────────────── #
#  Llançament de l’script Python                                                #
# ───────────────────────────────────────────────────────────────────────────── #
echo "=== Inici entrenament: $(date '+%Y-%m-%d %H:%M:%S') ==="
ulimit -n 16384
srun python MeteoGraphPC.py \
  --seq_dir      $SEQ_DIR \
  --batch_size   $BATCH_SIZE \
  --epochs       $EPOCHS \
  --lr           $LR \
  --hidden_dim   $HIDDEN_DIM \
  --model        $MODEL \
  --dl_num_workers $DL_NUM_WORKERS \
  --patience     $PATIENCE \
  --min_delta    $MIN_DELTA \
  --device       $DEVICE \
  --grad_clip    $GRAD_CLIP \
  --std_eps      $STD_EPS

echo "=== Fi entrenament:   $(date '+%Y-%m-%d %H:%M:%S') ==="
