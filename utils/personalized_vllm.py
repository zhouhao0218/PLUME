#!/usr/bin/env python3
# personalized_vllm.py
# Usage:
#   python personalized_vllm.py \
#       --model /path/to/llama-2-7b-chat-hf \
#       --data /path/to/data.jsonl \
#       --out_dir ./outputs \
#       --max_tokens 512
#
# Notes:
# - This uses vLLM (pip install vllm).
# - Your requested params: do_sample=False, temperature=0.0, top_k=10, top_p=1.
#   In vLLM, we set temperature=0.0 (greedy). top_k/top_p are kept but won’t affect greedy decoding.

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Tuple

import torch
from vllm import LLM, SamplingParams


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a dataset that is either:
       - JSONL (one JSON object per line), or
       - JSON array file, or
       - Single JSON object (wrapped into a list).
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    # Try JSONL first
    if "\n" in text:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        try:
            data = [json.loads(ln) for ln in lines]
            return data
        except json.JSONDecodeError:
            pass  # fall through to JSON
    # Try JSON
    js = json.loads(text)
    if isinstance(js, dict):
        return [js]
    if isinstance(js, list):
        return js
    raise ValueError("Unsupported dataset format: expected JSONL, JSON list, or JSON object.")


def make_prompt(item: Dict[str, Any]) -> str:
    """Build the prompt. Your dataset already embeds retrieved context into 'instruction'.
       We keep it as-is. If 'input' is non-empty, append it.
    """
    instr = item.get("instruction", "")
    inp = item.get("input", "")
    if inp:
        return f"{instr}\n{inp}"
    return instr


def get_ground_truth(item: Dict[str, Any]) -> str:
    """Extract ground truth from the dataset item and remove '### Response: ' prefix."""
    output = str(item.get("output", ""))
    # Remove "### Response: " prefix if present
    if output.startswith("### Response: "):
        output = output[len("### Response: "):]
    return output




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path or HF repo id for the LLaMA(-chat) model")
    ap.add_argument("--data", required=True, help="Path to dataset (json/jsonl)")
    ap.add_argument("--out_dir", required=True, help="Directory to save outputs")
    ap.add_argument("--max_tokens", type=int, default=512, help="Max new tokens to generate")
    ap.add_argument("--tensor_parallel_size", type=int, default=None, help="vLLM tensor parallel size (default: #GPUs)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed (irrelevant for greedy but kept for reproducibility)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    data = load_dataset(args.data)
    if len(data) == 0:
        print("Empty dataset.", file=sys.stderr)
        sys.exit(1)

    # Build prompts and ground truths
    prompts: List[str] = []
    ground_truths: List[str] = []
    for i, item in enumerate(data):
        prompts.append(make_prompt(item))
        ground_truths.append(get_ground_truth(item))

    # Initialize vLLM
    tps = args.tensor_parallel_size or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    print(f"Loading model: {args.model}  (tensor_parallel_size={tps})")
    llm = LLM(model=args.model, tensor_parallel_size=tps)

    # Sampling parameters (your requested settings)
    # Greedy decoding: temperature=0.0 automatically enables greedy decoding. top_k/top_p are kept but have no effect when temp=0.
    sampling_params = SamplingParams(
        temperature=0.,
        top_k=10,
        top_p=1.0,
        max_tokens=args.max_tokens,
        repetition_penalty=1.15, 
    )

    # Save prompts for debugging
    with open(os.path.join(args.out_dir, "prompts_debug.json"), "w", encoding="utf-8") as fdbg:
        json.dump([{"prompt": prompts[i], "ground_truth": ground_truths[i]} for i in range(len(prompts))], fdbg, ensure_ascii=False, indent=2)

    # Generate
    print(f"Generating {len(prompts)} responses ...")
    outputs = llm.generate(prompts, sampling_params)

    # Collect results
    results_path_jsonl = os.path.join(args.out_dir, "results.jsonl")
    results_path_json = os.path.join(args.out_dir, "results.json")
    results: List[Dict[str, Any]] = []

    for i, out in enumerate(outputs):
        # vLLM returns one RequestOutput per prompt; each has .outputs (list) with sampled candidates
        generated_text = out.outputs[0].text if out.outputs else ""
        rec = {
            "prompt": prompts[i],
            "output": ground_truths[i],  # Ground truth from dataset
            "generated_text": generated_text.strip(),  # LLM generated text
            "finish_reason": (out.outputs[0].finish_reason if out.outputs else "no_outputs"),
            "user_id": data[i].get("user_id", ""),  # Add user_id from original data
            "id": data[i].get("id", ""),  # Add id from original data
        }
        results.append(rec)

    # Save JSONL
    with open(results_path_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save JSON (array)
    with open(results_path_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved: {results_path_jsonl}")
    print(f"Saved: {results_path_json}")


if __name__ == "__main__":
    main()
