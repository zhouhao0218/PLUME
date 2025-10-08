#!/usr/bin/env bash
set -euo pipefail

############################
# Path Configuration (modify as needed)
############################
BASE_MODEL="output/abstract_generation_lora_mistral_7b_r64/ft"
OUTPUT_PATH="output/abstract_generation_mistral_7b/plume_share"

DATA_DIR="dataset/longlamp_abstract_generation"
TRAIN_FILE="$DATA_DIR/personalized_data_k1.jsonl"

# Your script actual locations
PLUME_SCRIPT="../../plume_resid_Share.py"

############################
# Plume1 training parameters
############################
# These parameters correspond to the CustomArgs class in plume1.py

############################
# 1) Train Plume1 (using plume1.py)
############################
echo "[Train] Start Plume1 training with plume1_resid_Share.py..."

python "$PLUME_SCRIPT" \
  --model_name_or_path "$BASE_MODEL" \
  --data_path "$TRAIN_FILE" \
  --output_dir "$OUTPUT_PATH" \
  --num_train_epochs 5 \
  --per_device_train_batch_size 3 \
  --learning_rate 1e-5 \
  --bf16 \
  --logging_steps 1 \
  --model_max_length 2048 \
  --lora_r 64 \
  --plume_mode phase1 \
  --shared_r 8 \
  --residue_rank 1 \
  --cu_coefficient 0.0 \

echo "[Train] Done. Results saved at: $OUTPUT_PATH"
