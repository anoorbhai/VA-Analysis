#!/usr/bin/env python3
"""
QLoRA Fine-tuning Script for Llama 3:8B on Verbal Autopsy Data

This script performs supervised fine-tuning using QLoRA (Quantized Low-Rank Adaptation)
on a locally downloaded Llama 3:8B model for verbal autopsy cause of death prediction.

QLoRA benefits:
- Significantly reduced memory usage through 4-bit quantization
- Maintains model quality through Low-Rank Adaptation
- Faster training and inference
- Suitable for single GPU training
"""

import os
import sys
import json
import logging
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)
import bitsandbytes as bnb

# Configure logging
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'finetune_qlora_{log_timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention for faster training"}
    )

@dataclass
class DataArguments:
    """Arguments for data configuration"""
    train_file: str = field(
        default="/home/noorbhaia/VA-Analysis/Fine-Tuning/train.jsonl",
        metadata={"help": "Path to training data file"}
    )
    test_file: str = field(
        default="/home/noorbhaia/VA-Analysis/Fine-Tuning/test.jsonl", 
        metadata={"help": "Path to test data file"}
    )
    max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length"}
    )

@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration"""
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA"}
    )

class VADataset:
    """Custom dataset handler for Verbal Autopsy data"""
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def load_and_process_data(self, file_path: str) -> Dataset:
        """Load and tokenize the JSONL data"""
        logger.info(f"Loading data from {file_path}")
        
        # Load JSONL file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        logger.info(f"Loaded {len(data)} examples")
        
        # Convert to dataset
        dataset = Dataset.from_list(data)
        
        # Tokenize the data
        logger.info("Tokenizing data...")
        tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        return tokenized_dataset
    
    def _tokenize_function(self, examples):
        """Tokenize the conversation format"""
        tokenized_inputs = []
        
        for messages in examples["messages"]:
            # Format the conversation
            conversation_text = self._format_conversation(messages)
            
            # Tokenize
            tokenized = self.tokenizer(
                conversation_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            tokenized_inputs.append(tokenized)
        
        # Batch the results
        batch = {}
        for key in tokenized_inputs[0].keys():
            batch[key] = [example[key] for example in tokenized_inputs]
            
        return batch
    
    def _format_conversation(self, messages: List[Dict]) -> str:
        """Format messages into a single conversation string"""
        conversation = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                conversation += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                conversation += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                conversation += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        
        return conversation

def setup_model_and_tokenizer(model_args: ModelArguments):
    """Setup model and tokenizer with QLoRA configuration"""
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    
    # BitsAndBytesConfig for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention else None
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora(model, lora_args: LoRAArguments):
    """Setup LoRA configuration"""
    logger.info("Setting up LoRA configuration")
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str
) -> Trainer:
    """Create and configure the trainer"""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        run_name=f"va_qlora_{log_timestamp}",
        seed=42,
        data_seed=42,
        group_by_length=True,
        length_column_name="length",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    return trainer

def main():
    """Main training function"""
    logger.info("Starting QLoRA fine-tuning for Verbal Autopsy")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. QLoRA requires GPU.")
        sys.exit(1)
    
    logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize arguments
    model_args = ModelArguments()
    data_args = DataArguments()
    lora_args = LoRAArguments()
    
    # Create output directory
    output_dir = Path(f"/home/noorbhaia/VA-Analysis/Fine-Tuning/models/va_qlora_{log_timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # Setup LoRA
    model = setup_lora(model, lora_args)
    
    # Load and process data
    dataset_handler = VADataset(tokenizer, data_args.max_length)
    train_dataset = dataset_handler.load_and_process_data(data_args.train_file)
    eval_dataset = dataset_handler.load_and_process_data(data_args.test_file)
    
    # Add length column for grouping
    def add_length(example):
        example["length"] = len(example["input_ids"])
        return example
    
    train_dataset = train_dataset.map(add_length)
    eval_dataset = eval_dataset.map(add_length)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=str(output_dir)
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    # Save training metrics
    metrics_file = output_dir / "training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(trainer.state.log_history, f, indent=2)
    
    logger.info(f"Training completed! Model saved to: {output_dir}")
    logger.info(f"Training metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()