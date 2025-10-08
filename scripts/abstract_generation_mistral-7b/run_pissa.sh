#!/usr/bin/env bash
set -euo pipefail

############################
# Path Configuration (modify as needed)
############################
BASE_MODEL="models/Mistral-7B-Instruct-v0.2"
RES_MODEL="output/Mistral-7B-Instruct-v0.2-pissa"
OUTPUT_PATH="output/abstract_generation_pissa_mistral_7b_r64"

DATA_DIR="dataset/longlamp_abstract_generation"
TRAIN_FILE="$DATA_DIR/train_k1_subset_200_users.jsonl"
TEST_FILE="$DATA_DIR/test_k1_subset_200_users.jsonl"

# Your script actual locations
TRAIN_SCRIPT="../../train_lora.py"
INFER_SCRIPT="../../utils/personalized_vllm.py"
EVAL_SCRIPT="../../utils/personalized_eval.py"

# Inference & evaluation output
INFER_OUT="results/pissa"
EVAL_NAME="pissa_mistral_7b_k1_abstract_generation" 

############################
# 1) Train PiSSA (merge weights)
############################
echo "[Train] Start PiSSA training..."
python "$TRAIN_SCRIPT" \
  --model_name_or_path "$RES_MODEL" \
  --full_finetune False \
  --bf16 \
  --adapter_name_or_path "pissa_init" \
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
  --logging_steps 50 \
  --lr_scheduler_type cosine \
  --merge True

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
