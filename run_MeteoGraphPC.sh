#!/bin/bash
# Script per entrenar MeteoGraphPC directament en un node de càlcul sense SLURM

set -euo pipefail
#set -x
echo ">>> Script iniciat correctament"

# ───────────────────────────────────────────────────────────── #
# LÍMIT DE RAM: 100GB (ajustable segons la disponibilitat i el que vols usar)
ulimit -v $((100 * 1024 * 1024))

# ───────────────────────────────────────────────────────────── #
# CONFIGURACIÓ MANUAL DE LA GPU
export CUDA_VISIBLE_DEVICES=1     # <-- Canvia-ho segons quina GPU vulguis usar!

# ───────────────────────────────────────────────────────────── #
# ACTIVACIÓ DE L'ENTORN CONDA
#source ~/miniconda3/etc/profile.d/conda.sh
#set +u
#conda activate meteographpc
#set -u

# ───────────────────────────────────────────────────────────── #
# LIMITACIÓ DE RECURSOS CPU/RAM
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ───────────────────────────────────────────────────────────── #
# DIRECTORI DE LOGS
mkdir -p logs

# ───────────────────────────────────────────────────────────── #
# PARÀMETRES D'ENTRENAMENT
SEQ_DIR="/data2/users/nfarres/All_Sequences_ws48_str12_hh6_chunksde50"
BATCH_SIZE=8
EPOCHS=20
LR=1e-5
LR_SCHEDULER="onecycle"
HIDDEN_DIM=128
PATIENCE=8
MIN_DELTA=1e-4
DEVICE="cuda"
GRAD_CLIP=1.0
STD_EPS=1e-6
DL_NUM_WORKERS=2        
MODEL="MeteoGraphPC_v1" (en aquest treball s'ha fet ús del model MeteoGraphPC_v1; les altres versions estan en fase de proves i no s'han provat ni utilitzat en aquest treball)
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

WS=48 # window size
STR=12 # stride
HH=6  # horitzó predicció

#Modificar segons faci falta
SAVE_DIR="/data2/users/nfarres/checkpoints/${MODEL}_ws${WS}_str${STR}_hh${HH}_${TIMESTAMP}"
LOG_CSV="/data2/users/nfarres/checkpoints/${MODEL}_ws${WS}_str${STR}_hh${HH}_${TIMESTAMP}/train_${MODEL}_ws${WS}_str${STR}_hh${HH}_${TIMESTAMP}.csv"
NORM_JSON="PC_norm_params.json"

# ───────────────────────────────────────────────────────────── #
# EXECUCIÓ DEL MODEL
echo "=== Inici entrenament: $(date '+%Y-%m-%d %H:%M:%S') ==="
python MeteoGraphPC.py \
  --seq_dir      $SEQ_DIR \
  --batch_size   $BATCH_SIZE \
  --epochs       $EPOCHS \
  --lr           $LR \
  --lr_scheduler $LR_SCHEDULER \
  --hidden_dim   $HIDDEN_DIM \
  --model        $MODEL \
  --input_indices  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16 \
  --target_indices 0  1  2  3  4  15 16 \
  --use_edge_attr \
  --use_mask \
  --dl_num_workers $DL_NUM_WORKERS \
  --patience     $PATIENCE \
  --min_delta    $MIN_DELTA \
  --device       $DEVICE \
  --grad_clip    $GRAD_CLIP \
  --std_eps      $STD_EPS \
  --save_dir     $SAVE_DIR \
  --log_csv      $LOG_CSV \
  --norm_json    $NORM_JSON \
  | tee logs/train_${MODEL}_ws${WS}_str${STR}_hh${HH}_${TIMESTAMP}.log

echo "=== Fi entrenament:   $(date '+%Y-%m-%d %H:%M:%S') ==="
