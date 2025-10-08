#!/usr/bin/env bash
set -euo pipefail

############################
# Path Configuration (modify as needed)
############################
BASE_MODEL="models/Mistral-7B-Instruct-v0.2"
OUTPUT_PATH="output/abstract_generation_qlora_mistral_7b_r64"

DATA_DIR="dataset/longlamp_abstract_generation"
TRAIN_FILE="$DATA_DIR/train_k1_subset_200_users.jsonl"
TEST_FILE="$DATA_DIR/test_k1_subset_200_users.jsonl"

# Your script actual locations
TRAIN_SCRIPT="../../train_lora.py"
INFER_SCRIPT="../../utils/personalized_vllm.py"
EVAL_SCRIPT="../../utils/personalized_eval.py"

# Inference & evaluation output
INFER_OUT="results/qlora"
EVAL_NAME="qlora_mistral_7b_k1_abstract_generation_r64"

############################
# 1) Train QLoRA (4-bit quantization)
############################
echo "[Train] Start QLoRA training with 4-bit quantization..."
python "$TRAIN_SCRIPT" \
  --model_name_or_path "$BASE_MODEL" \
  --full_finetune False \
  --do_train \
  --bf16 \
  --bits 4 \
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
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 128 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --logging_steps 1 \
  --lr_scheduler_type cosine \
  --merge True

echo "[Train] Done. Merged model saved at: $OUTPUT_PATH/ft_merged"

############################
# 2) Model merge (merge LoRA adapter to base model)
############################
ADAPTER_PATH="$OUTPUT_PATH/ft"
MERGED_MODEL_PATH="$OUTPUT_PATH/ft_merged_manual"
MERGE_SCRIPT="../../utils/merge_adapter.py"

echo "[Merge] Starting model merge..."
echo "[Merge] Base model: $BASE_MODEL"
echo "[Merge] Adapter: $ADAPTER_PATH"
echo "[Merge] Output: $MERGED_MODEL_PATH"

# Check if files exist
if [[ ! -d "$BASE_MODEL" ]]; then
    echo "[Error] Base model not found: $BASE_MODEL"
    exit 1
fi

if [[ ! -d "$ADAPTER_PATH" ]]; then
    echo "[Error] Adapter not found: $ADAPTER_PATH"
    exit 1
fi

if [[ ! -f "$MERGE_SCRIPT" ]]; then
    echo "[Error] Merge script not found: $MERGE_SCRIPT"
    exit 1
fi

# Create output directory
mkdir -p "$MERGED_MODEL_PATH"

# Set environment variable to ensure CPU usage for merging
export CUDA_VISIBLE_DEVICES=""

echo "[Merge] This process runs on CPU and may take several minutes..."

python "$MERGE_SCRIPT" \
  --base_model "$BASE_MODEL" \
  --adapter "$ADAPTER_PATH" \
  --output_path "$MERGED_MODEL_PATH"

# Verify merge results
echo "[Verify] Checking merged model..."

MODEL_FOUND=false
CONFIG_FOUND=false

# Check model files
if [[ -f "$MERGED_MODEL_PATH/pytorch_model.bin" ]] || \
   [[ -f "$MERGED_MODEL_PATH/model.safetensors" ]] || \
   ls "$MERGED_MODEL_PATH"/model-*.safetensors 1> /dev/null 2>&1; then
    MODEL_FOUND=true
fi

# Check config files
if [[ -f "$MERGED_MODEL_PATH/config.json" ]] && [[ -f "$MERGED_MODEL_PATH/tokenizer.json" ]]; then
    CONFIG_FOUND=true
fi

if [[ "$MODEL_FOUND" == true ]] && [[ "$CONFIG_FOUND" == true ]]; then
    echo "[Success] Model merged successfully!"
    echo "[Success] Merged model saved at: $MERGED_MODEL_PATH"
    
    # Show file sizes
    echo "[Info] Model files:"
    ls -lh "$MERGED_MODEL_PATH"/*.bin "$MERGED_MODEL_PATH"/*.safetensors 2>/dev/null || true
    echo "[Info] Config files:"
    ls -lh "$MERGED_MODEL_PATH"/*.json 2>/dev/null || true
    
    # Show total size
    TOTAL_SIZE=$(du -sh "$MERGED_MODEL_PATH" | cut -f1)
    echo "[Info] Total model size: $TOTAL_SIZE"
else
    echo "[Error] Model merge failed - required files not found"
    echo "[Error] Model files found: $MODEL_FOUND"
    echo "[Error] Config files found: $CONFIG_FOUND"
    exit 1
fi

echo "[Done] Model merge completed successfully!"

# Reset CUDA_VISIBLE_DEVICES for GPU inference
unset CUDA_VISIBLE_DEVICES

############################
# 3) vLLM inference (using merged model)
############################
mkdir -p "$INFER_OUT"
echo "[Infer] Using model: $OUTPUT_PATH/ft_merged_manual"
echo "[Infer] Test data:   $TEST_FILE"

python "$INFER_SCRIPT" \
  --model "$OUTPUT_PATH/ft_merged_manual" \
  --data "$TEST_FILE" \
  --out_dir "$INFER_OUT" \
  --max_tokens 512
# To limit parallel GPUs: set CUDA_VISIBLE_DEVICES=0 before the above command or add --tensor_parallel_size in the script

echo "[Infer] Results saved under: $INFER_OUT"

############################
# 4) Evaluation
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
