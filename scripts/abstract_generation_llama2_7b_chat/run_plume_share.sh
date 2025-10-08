#!/usr/bin/env bash
set -euo pipefail

############################
# Path Configuration (modify as needed)
############################
BASE_MODEL="output/abstract_generation_lora_llama2_7b_r64/ft"
OUTPUT_PATH="output/abstract_generation_llama2_7b_chat/plume_share"

DATA_DIR="dataset/longlamp_abstract_generation"
TRAIN_FILE="$DATA_DIR/personalized_data_k1.jsonl"

# Your script actual locations
PLUME_SCRIPT="../../plume1_residHire_llama.py"

############################
# 1) Train Plume1 (using plume1.py)
############################
echo "[Train] Start Plume1 training with plume1_residHire_llama.py..."

python "$PLUME_SCRIPT" \
  --model_name_or_path "$BASE_MODEL" \
  --data_path "$TRAIN_FILE" \
  --output_dir "$OUTPUT_PATH" \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --learning_rate 1e-5 \
  --bf16 \
  --logging_steps 1 \
  --model_max_length 2048 \
  --lora_r 64 \
  --plume_mode phase1 \
  --shared_r 64 \
  --residue_rank 1 \

echo "[Train] Done. Results saved at: $OUTPUT_PATH"
