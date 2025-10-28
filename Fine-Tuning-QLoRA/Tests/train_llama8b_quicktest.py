# /spaces/va-qlora/scripts/train_qlora_sft.py
import os, json, torch
from dataclasses import dataclass
from typing import Dict, List
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face token from environment
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_HUB_TOKEN not found in environment variables. Please set it in .env file.")

login(token=hf_token)

# ----------------- Config - QUICK TEST VERSION -----------------
MODEL_DIR          = "meta-llama/Llama-3.1-8B-Instruct"
TRAIN_PATH         = "/spaces/25G05/Fine-Tuning/train.jsonl"
VAL_PATH           = "/spaces/25G05/Fine-Tuning/test.jsonl"
OUTPUT_DIR         = "Fine-Tuning-QLoRA/Tests/llama3-8b-qlora-va-quicktest"
MAX_INPUT_TOKENS   = 512          # Reduced from 2048
BATCH_SIZE_PER_DEV = 1            # Reduced from 2
GR_ACCUM_STEPS     = 2            # Reduced from 8
LR                 = 2e-4
NUM_EPOCHS         = 1            # Reduced from 3 - just 1 epoch for testing
MAX_TRAIN_SAMPLES  = 10           # Only use 10 training samples
MAX_VAL_SAMPLES    = 5            # Only use 5 validation samples
LORA_R             = 8            # Reduced from 16
LORA_ALPHA         = 16           # Reduced from 32
LORA_DROPOUT       = 0.05
TARGET_MODULES     = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]  # common for LLaMA
# ------------------------------------------

def format_example(ex):
    """
    Convert chat format to single text string for SFTTrainer.
    The dataset has 'messages' with system, user, and assistant messages.
    """
    messages = ex["messages"]
    
    # Build conversation text
    conversation = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            conversation += f"<|system|>\n{content}\n"
        elif role == "user":
            conversation += f"<|user|>\n{content}\n"
        elif role == "assistant":
            conversation += f"<|assistant|>\n{content}\n"
    
    return {"text": conversation}


def tokenize(tokenizer, text: str):
    return tokenizer(text, max_length=MAX_INPUT_TOKENS, truncation=True, padding="max_length")

def main():
    print("=" * 50)
    print("QUICK TEST MODE - Limited samples and epochs")
    print("=" * 50)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit load (QLoRA)
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Loading model in 4-bit mode with {compute_dtype} compute dtype...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA
    peft_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    print(f"LoRA config: r={LORA_R}, alpha={LORA_ALPHA}")

    # Data
    print(f"Loading datasets from {TRAIN_PATH} and {VAL_PATH}...")
    dataset = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": VAL_PATH})
    
    # Limit dataset size for quick testing
    dataset["train"] = dataset["train"].select(range(min(MAX_TRAIN_SAMPLES, len(dataset["train"]))))
    dataset["validation"] = dataset["validation"].select(range(min(MAX_VAL_SAMPLES, len(dataset["validation"]))))
    
    print(f"Using {len(dataset['train'])} training samples and {len(dataset['validation'])} validation samples")
    
    def map_fn(ex):
        formatted = format_example(ex)  # Returns {"text": "..."}
        text = formatted["text"]  # Extract the string
        tokens = tokenize(tokenizer, text)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(map_fn, remove_columns=dataset["train"].column_names, num_proc=4)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE_PER_DEV,
        per_device_eval_batch_size=BATCH_SIZE_PER_DEV,
        gradient_accumulation_steps=GR_ACCUM_STEPS,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),  
        logging_steps=10,                # Log more frequently
        save_steps=50,                   # Save less frequently for quick test
        evaluation_strategy="steps",
        eval_steps=50,                   # Eval less frequently
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        optim="paged_adamw_8bit",  
        report_to=[],               
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    
    # Save only the LoRA adapter (tiny)
    adapter_path = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("=" * 50)
    print(f"Quick test completed! LoRA adapter saved to: {adapter_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()
