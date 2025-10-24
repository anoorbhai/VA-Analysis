#!/usr/bin/env python3
"""
Configuration script for QLoRA fine-tuning
Allows easy modification of training parameters without editing the main script
"""

from dataclasses import dataclass, field
from typing import List
import json

@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning"""
    
    # Model configuration
    model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    use_flash_attention: bool = True
    
    # Data configuration
    train_file: str = "/home/noorbhaia/VA-Analysis/Fine-Tuning/train.jsonl"
    test_file: str = "/home/noorbhaia/VA-Analysis/Fine-Tuning/test.jsonl"
    max_length: int = 4096
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training configuration
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Logging and evaluation
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # System configuration
    fp16: bool = True
    dataloader_pin_memory: bool = False
    group_by_length: bool = True
    seed: int = 42

def save_config(config: QLoRAConfig, filepath: str):
    """Save configuration to JSON file"""
    config_dict = {
        'model_name_or_path': config.model_name_or_path,
        'use_flash_attention': config.use_flash_attention,
        'train_file': config.train_file,
        'test_file': config.test_file,
        'max_length': config.max_length,
        'lora_r': config.lora_r,
        'lora_alpha': config.lora_alpha,
        'lora_dropout': config.lora_dropout,
        'target_modules': config.target_modules,
        'per_device_train_batch_size': config.per_device_train_batch_size,
        'per_device_eval_batch_size': config.per_device_eval_batch_size,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'num_train_epochs': config.num_train_epochs,
        'learning_rate': config.learning_rate,
        'lr_scheduler_type': config.lr_scheduler_type,
        'warmup_ratio': config.warmup_ratio,
        'weight_decay': config.weight_decay,
        'logging_steps': config.logging_steps,
        'save_steps': config.save_steps,
        'eval_steps': config.eval_steps,
        'save_total_limit': config.save_total_limit,
        'fp16': config.fp16,
        'dataloader_pin_memory': config.dataloader_pin_memory,
        'group_by_length': config.group_by_length,
        'seed': config.seed
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_config(filepath: str) -> QLoRAConfig:
    """Load configuration from JSON file"""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    return QLoRAConfig(**config_dict)

# Predefined configurations for different scenarios

def get_quick_test_config() -> QLoRAConfig:
    """Configuration for quick testing (small model, fewer epochs)"""
    config = QLoRAConfig()
    config.num_train_epochs = 1
    config.save_steps = 100
    config.eval_steps = 100
    config.max_length = 2048
    return config

def get_production_config() -> QLoRAConfig:
    """Configuration for production training"""
    config = QLoRAConfig()
    config.num_train_epochs = 5
    config.learning_rate = 1e-4
    config.save_steps = 250
    config.eval_steps = 250
    config.lora_r = 32
    config.lora_alpha = 64
    return config

def get_memory_optimized_config() -> QLoRAConfig:
    """Configuration for limited GPU memory"""
    config = QLoRAConfig()
    config.per_device_train_batch_size = 1
    config.gradient_accumulation_steps = 8
    config.max_length = 2048
    config.fp16 = True
    return config

if __name__ == "__main__":
    # Generate default config files
    default_config = QLoRAConfig()
    save_config(default_config, "config_default.json")
    
    quick_config = get_quick_test_config()
    save_config(quick_config, "config_quick_test.json")
    
    prod_config = get_production_config()
    save_config(prod_config, "config_production.json")
    
    memory_config = get_memory_optimized_config()
    save_config(memory_config, "config_memory_optimized.json")
    
    print("Generated configuration files:")
    print("- config_default.json")
    print("- config_quick_test.json") 
    print("- config_production.json")
    print("- config_memory_optimized.json")