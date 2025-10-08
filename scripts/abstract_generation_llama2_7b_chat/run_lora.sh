#!/usr/bin/env bash
set -euo pipefail

############################
# Path Configuration (modify as needed)
############################
BASE_MODEL="models/llama-2-7b-chat-hf"
OUTPUT_PATH="output/abstract_generation_lora_llama2_7b_r64"

DATA_DIR="dataset/longlamp_abstract_generation"
TRAIN_FILE="$DATA_DIR/train_k1_subset_200_users.jsonl"
TEST_FILE="$DATA_DIR/test_k1_subset_200_users.jsonl"

# Your script actual locations
TRAIN_SCRIPT="../../train_personalized.py"
INFER_SCRIPT="../../utils/personalized_vllm.py"
EVAL_SCRIPT="../../utils/personalized_eval.py"

# Inference & evaluation output
INFER_OUT="$OUTPUT_PATH/infer_k1"
EVAL_NAME="lora_llama2_chat_k1_abstract_generation"

############################
# 1) Train LoRA (merge weights)
############################
echo "[Train] Start LoRA training..."
python "$TRAIN_SCRIPT" \
  --model_name_or_path "$BASE_MODEL" \
  --full_finetune False \
  --do_train \
  --bf16 \
  --init_weights True \
  --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
  --lora_rank 64 \
  --lora_alpha 64 \
  --lora_dropout 0 \
  --data_path "$TRAIN_FILE" \
  --sub_task main \
  --dataset_split train \
  --dataset_field instruction output \
  --output_dir "$OUTPUT_PATH" \
  --num_train_epochs 5 \
  --model_max_length 1024 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --logging_steps 1 \
  --lr_scheduler_type cosine \
  --merge True

echo "[Train] Done. Merged model at: $OUTPUT_PATH"

############################
# 2) vLLM inference (using merged model)
############################
mkdir -p "$INFER_OUT"
echo "[Infer] Using model: $OUTPUT_PATH/ft_merged"
echo "[Infer] Test data:   $TEST_FILE"

python "$INFER_SCRIPT" \
  --model "$OUTPUT_PATH/ft_merged" \
  --data "$TEST_FILE" \
  --out_dir "$INFER_OUT" \
  --max_tokens 512
# To limit parallel GPUs: set CUDA_VISIBLE_DEVICES=0 before the above command or add --tensor_parallel_size in the script

echo "[Infer] Results saved under: $INFER_OUT"

############################
# 3) Evaluation
############################
RESULTS_JSON="$INFER_OUT/results.json"
if [[ -f "$RESULTS_JSON" ]]; then
  echo "[Eval] Evaluating: $RESULTS_JSON"
  python "$EVAL_SCRIPT" \
    --data "$RESULTS_JSON" \
    --name "$EVAL_NAME"
  echo "[Eval] Done. See metrics.txt in: $INFER_OUT"
else
  echo "[Eval] WARNING: $RESULTS_JSON not found; skip evaluation."
fi
