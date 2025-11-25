import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Sequence, Optional
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from datasets import Dataset
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
    PreTrainedTokenizer,
    LlamaTokenizer,
    StoppingCriteria, StoppingCriteriaList,
)
from transformers import logging as hf_logging
from peft import LoraConfig, get_peft_model, PeftModel

# Import evaluation metrics
from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score

class PlumeAB_share(nn.Module):

    def __init__(self, base_linear: nn.Linear, A, B, r, residue_rank: int = 1, cu_coefficient: float = 2.0, resid_coefficient: float = 1.0):
        super().__init__()
        self.base_linear = base_linear
        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.A = nn.Parameter(A, requires_grad=False)
        self.B = nn.Parameter(B, requires_grad=False)

        self.r = r
        self.residue_rank = residue_rank
        self.cu_coefficient = cu_coefficient
        self.resid_coefficient = resid_coefficient
        self.phase = "full"

        self.Cu = None
        self.Au = None
        self.Bu = None

        self._cached_W_user = None

    def inject_Cu(self):
        device = self.base_linear.weight.device

        self.Cu = nn.Parameter(torch.zeros(self.r, self.r, device=device, requires_grad=True))
        out_features, in_features = self.base_linear.weight.shape
        self.Au = nn.Parameter(torch.zeros(out_features, self.residue_rank, device=device, requires_grad=True))
        self.Bu = nn.Parameter(torch.empty(in_features, self.residue_rank, device=device, requires_grad=True))
        nn.init.kaiming_uniform_(self.Bu, a=math.sqrt(5))


        self.phase = "full"
        self._cached_W_user = None

    def inject_qa_qb(self, qa, qb):
        self.qa = qa
        self.qb = qb

    def forward(self, x):
        if self.training or self._cached_W_user is None:
            if self.phase == "full":
                assert self.Cu is not None
                device = self.base_linear.weight.device
                I = torch.eye(self.r).to(x.dtype).to(device)
                W_low = self.B @ (I + self.cu_coefficient * self.Cu) @ self.A   # (out, r) @ (r, r) @ (r, in) -> (out, in)    
                W_resid = self.resid_coefficient * (self.Au) @ (self.Bu.T)                       
                W_low += W_resid
                W_low += self.qb.T @ self.qa.T
                
            else:
                raise ValueError(f"Unknown phase: {self.phase}")

            W_user = self.base_linear.weight + W_low
            if not self.training:
                self._cached_W_user = W_user.detach().to(x.dtype)

            return F.linear(x, W_user.to(x.dtype), self.base_linear.bias)
        return F.linear(x, self._cached_W_user, self.base_linear.bias)


IGNORE_INDEX = -100
PLUME_LAYER_CLS = PlumeAB_share

# Download NLTK data for METEOR
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


@dataclass
class CustomArgs(TrainingArguments):
    output_dir: str = field(default="./outputs")
    per_device_train_batch_size: int = field(default=2)
    num_train_epochs: int = field(default=2)
    learning_rate: float = field(default=1e-4)
    bf16: bool = field(default=True)
    logging_steps: int = field(default=20)
    save_strategy: str = field(default="no")
    report_to: str = field(default="none")
    remove_unused_columns: bool = field(default=False)

    model_max_length: int = field(default=2048)
    lora_r: int = field(default=64) 
    seed: int = field(default=42)
    data_path: str = field(default=None, metadata={"help": "Path to input data."})
    model_name_or_path: str = field(default=None, metadata={"help": "Path or name of base model."})
    
    note: str = field(default="", metadata={"help": "Optional note to include in log file name."})
    plume_mode: str = field(default="phase1", metadata={"help": "phase1 or phase2 or phase3 or full"})
    shared_r: int = field(default=8, metadata={"help": "Shared rank for plume layers"})
    residue_rank: int = field(default=1, metadata={"help": "Rank for residue parameters Au and Bu"})
    max_new_tokens: int = field(default=600, metadata={"help": "Maximum number of new tokens to generate"})
    cu_coefficient: float = field(default=2.0, metadata={"help": "Coefficient for Cu in the formula: I + cu_coefficient * Cu"})
    resid_coefficient: float = field(default=1.0, metadata={"help": "Coefficient for residue parameters Au and Bu"})




import datetime

def setup_logger(model_name: str, data_name: str, note: str = "", log_dir: str = "./saved_logs") -> logging.Logger:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    formatter.converter = time.localtime

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #model_tag = model_name.replace("/", "_")
    model_tag = "oppu"
    data_tag = data_name.replace("/", "_")
    note_tag = note.replace(" ", "_") if note else "nonote"

    full_log_dir = os.path.join(log_dir, data_tag, model_tag)
    os.makedirs(full_log_dir, exist_ok=True)

    # 👇 note 加在文件名末尾
    log_path = os.path.join(full_log_dir, f"{date_str}_{note_tag}.log")

    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    print(f"Logging to file: {log_path}")
    return logger

def redirect_transformers_logging_to(logger: logging.Logger):
    hf_logger = hf_logging.get_logger()
    hf_logger.handlers = []  
    for handler in logger.handlers:
        hf_logger.addHandler(handler) 
    hf_logger.setLevel(logger.level)


def evaluate_text_generation(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Evaluate text generation using ROUGE-1, ROUGE-L, and METEOR metrics.
    
    Args:
        predictions: List of generated texts
        references: List of reference texts
    
    Returns:
        Dictionary containing ROUGE-1, ROUGE-L, and METEOR scores
    """
    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rougeL_scores = []
    meteor_scores = []
    
    for pred, ref in zip(predictions, references):
        # Calculate ROUGE scores
        rouge_scores = rouge.score(ref, pred)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
        
        # Calculate METEOR score
        # Tokenize for METEOR
        pred_tokens = nltk.word_tokenize(pred.lower())
        ref_tokens = nltk.word_tokenize(ref.lower())
        meteor_scores.append(meteor_score([ref_tokens], pred_tokens))
    
    return {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores),
        'rougeL': sum(rougeL_scores) / len(rougeL_scores),
        'meteor': sum(meteor_scores) / len(meteor_scores)
    }


def tokenize_pair(sources: Sequence[str], targets: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    input_ids = []
    labels = []
    for src, ex in zip(sources, examples):
        tok_ex = tokenizer(
            ex,
            max_length=tokenizer.model_max_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        tok_src = tokenizer(
            src,
            max_length=tokenizer.model_max_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        ids = tok_ex["input_ids"][0]
        label = ids.clone()
        label[:tok_src["input_ids"].shape[1]] = IGNORE_INDEX
        input_ids.append(ids)
        labels.append(label)
    return {"input_ids": input_ids, "labels": labels}


class SupervisedDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ex["input_ids"]) for ex in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ex["labels"]) for ex in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id)
        }

class StopOnSubstrings(StoppingCriteria):
    def __init__(self, stop_strs: List[str], tokenizer: PreTrainedTokenizer):
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strs]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_sequence in self.stop_ids:
            if input_ids.shape[1] >= len(stop_sequence):
                if input_ids[0, -len(stop_sequence):].tolist() == stop_sequence:
                    return True
        return False

def get_svd_cache_dir(model_path: str, root_dir="./saved_cache/svd_cache"):

    norm_path = os.path.normpath(model_path)
    svd_dir = os.path.join(root_dir, norm_path)
    os.makedirs(svd_dir, exist_ok=True)
    return svd_dir

def create_trainer(model, tokenizer, args, train_dataset):
    return Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        data_collator=SupervisedDataCollator(tokenizer)
    )

def load_aplaud_params(user_id: str, model: nn.Module, args, phase: str) -> bool:
    return False

def replace_with_plume_from_peft(peft_model, r, residue_rank=1, svd_cache_dir=None, cu_coefficient=2.0, resid_coefficient=1.0):
    to_replace = []

    for name, module in peft_model.named_modules():
        if (
            hasattr(module, "lora_A") and
            hasattr(module, "lora_B") and
            not isinstance(module, PLUME_LAYER_CLS)
        ):
            to_replace.append((name, module))


    for name, module in to_replace:
        A = module.lora_A["default"].weight.data.float().to("cuda")
        B = module.lora_B["default"].weight.data.float().to("cuda")

        base_linear = nn.Linear(module.in_features, module.out_features, bias=False).to("cuda")
        base_linear.weight.data.copy_(module.base_layer.weight.data)

        plumer = PLUME_LAYER_CLS(base_linear, A, B, r=r, residue_rank=residue_rank, cu_coefficient=cu_coefficient, resid_coefficient=resid_coefficient)
        
        parent = peft_model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], plumer)



def main():
    parser = transformers.HfArgumentParser(CustomArgs)
    args, = parser.parse_args_into_dataclasses()
    logger = setup_logger(
        model_name=args.model_name_or_path,
        data_name=args.data_path,
        note=args.note
    )
    redirect_transformers_logging_to(logger)

    print("Start training...")
    logger.setLevel(logging.INFO)

    transformers.logging.set_verbosity_error() 


    set_seed(args.seed)

    tokenizer = LlamaTokenizer.from_pretrained(
        "/users/PGS0218/hzhou6/LoRA/models/Mistral-7B-Instruct-v0.2",
        padding_side="right",
        trust_remote_code=True
    )
    tokenizer.model_max_length = args.model_max_length
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, args.model_name_or_path)

    svd_cache_dir = get_svd_cache_dir(args.model_name_or_path)
    replace_with_plume_from_peft(model, r=args.lora_r, residue_rank=args.residue_rank, svd_cache_dir=svd_cache_dir, cu_coefficient=args.cu_coefficient, resid_coefficient=args.resid_coefficient)


    with open(args.data_path) as f:
        user_data_list = [json.loads(line) for line in f]

    all_preds = []


    for user in tqdm(user_data_list[:200], desc="Per-user training", leave=False):
        print(f"Training for user: {user['user_id']}")

        sources = [item["instruction"] for item in user["train"]]
        targets = [item["output"] + tokenizer.eos_token for item in user["train"]]
        if not sources:
            continue

        tokenized = tokenize_pair(sources, targets, tokenizer)
        train_dataset = Dataset.from_list([
            {"input_ids": x, "labels": y}
            for x, y in zip(tokenized["input_ids"], tokenized["labels"])
        ])

        model.q_a = nn.Parameter(torch.zeros(4096, args.shared_r))
        model.q_b = nn.Parameter(torch.empty(args.shared_r, 4096))
        nn.init.normal_(model.q_b)
        model.k_a = nn.Parameter(torch.zeros(4096, args.shared_r))
        model.k_b = nn.Parameter(torch.empty(args.shared_r, 1024))
        nn.init.normal_(model.k_b)
        model.v_a = nn.Parameter(torch.zeros(4096, args.shared_r))
        model.v_b = nn.Parameter(torch.empty(args.shared_r, 1024))
        nn.init.normal_(model.v_b)
        model.o_a = nn.Parameter(torch.zeros(4096, args.shared_r))
        model.o_b = nn.Parameter(torch.empty(args.shared_r, 4096))
        nn.init.normal_(model.o_b)
        model.up_a = nn.Parameter(torch.zeros(4096, args.shared_r))
        model.up_b = nn.Parameter(torch.empty(args.shared_r, 14336))
        nn.init.normal_(model.up_b)
        model.down_a = nn.Parameter(torch.zeros(14336, args.shared_r))
        model.down_b = nn.Parameter(torch.empty(args.shared_r, 4096))
        nn.init.normal_(model.down_b)
        model.gate_a = nn.Parameter(torch.zeros(4096, args.shared_r))
        model.gate_b = nn.Parameter(torch.empty(args.shared_r, 14336))
        nn.init.normal_(model.gate_b)

        for mod_name, module in model.named_modules():
            if isinstance(module, PLUME_LAYER_CLS):
                if mod_name.endswith("q_proj"):
                    module.inject_qa_qb(model.q_a, model.q_b)
                elif mod_name.endswith("k_proj"):
                    module.inject_qa_qb(model.k_a, model.k_b)
                elif mod_name.endswith("v_proj"):
                    module.inject_qa_qb(model.v_a, model.v_b)
                elif mod_name.endswith("o_proj"):
                    module.inject_qa_qb(model.o_a, model.o_b)
                elif mod_name.endswith("up_proj"):
                    module.inject_qa_qb(model.up_a, model.up_b)
                elif mod_name.endswith("down_proj"):
                    module.inject_qa_qb(model.down_a, model.down_b)
                elif mod_name.endswith("gate_proj"):
                    module.inject_qa_qb(model.gate_a, model.gate_b)



        for module in model.modules():
            if isinstance(module, PLUME_LAYER_CLS):
                module.inject_Cu()

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            data_collator=SupervisedDataCollator(tokenizer)
        )
       
        
        # === Phase 1 ===
        if args.plume_mode in ["phase1", "phase2", "phase3", "full"]:
            if not load_aplaud_params(user['user_id'], model, args, phase="phase1"):
                for module in model.modules():
                    if isinstance(module, PLUME_LAYER_CLS):
                        module.inject_Cu()

                model.train()
                trainer.train()


        print("Eval for user: %s", user['user_id'])
        model.eval()
        test_prompts = [q["input"] for q in user["test"]]
        test_golds = [q["gold"] for q in user["test"]]
        test_ids = [q["id"] for q in user["test"]]

        preds = []
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad(), torch.autocast("cuda"):
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=0.,
                    top_p=1.0,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=1.15,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([
                        StopOnSubstrings(["Instruction:", "Response:", "Instruction", "Response"], tokenizer)
                    ])
                )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = decoded[len(prompt):].strip()
            preds.append(pred)
            print('pred:--------------------------------')
            print(pred)

        for qid, pred, gold in zip(test_ids, preds, test_golds):
            all_preds.append({"id": qid, "user_id": user['user_id'], "output": pred, "gold": gold})


    

    # Final evaluation using ROUGE and METEOR metrics
    predictions = [x["output"] for x in all_preds]
    references = [x["gold"] for x in all_preds]
    
    print(f"Evaluating {len(predictions)} predictions...")
    
    # Calculate evaluation metrics
    metrics = evaluate_text_generation(predictions, references)
    
    print(f"\n✅ Final Evaluation:")
    print(f"ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"ROUGE-L: {metrics['rougeL']:.4f}")
    print(f"METEOR: {metrics['meteor']:.4f}")
    
    # Log some example predictions for inspection
    print(f"\n📝 Sample Predictions:")
    for i, (pred, ref) in enumerate(zip(predictions[:3], references[:3])):
        print(f"Example {i+1}:")
        print(f"Reference: {ref[:200]}...")
        print(f"Prediction: {pred[:200]}...")
        print("-" * 50)
    
    # Save all predictions to JSON file
    os.makedirs(args.output_dir, exist_ok=True)
    predictions_file = os.path.join(args.output_dir, "all_predictions.json")
    
    # Create a comprehensive results dictionary
    results = {
        "evaluation_metrics": metrics,
        "predictions": all_preds,
        "summary": {
            "total_predictions": len(all_preds),
            "model_name": args.model_name_or_path,
            "data_path": args.data_path,
            "lora_r": args.lora_r,
            "seed": args.seed,
            "note": args.note,
            "plume_mode": args.plume_mode,
            "shared_r": args.shared_r,
            "residue_rank": args.residue_rank
        }
    }
    
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 All predictions saved to: {predictions_file}")
    print(f"Total predictions saved: {len(all_preds)}")


if __name__ == "__main__":
    main()
