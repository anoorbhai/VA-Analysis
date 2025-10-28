#!/bin/bash
#SBATCH --job-name=train_llama3
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_8b_%j.out
#SBATCH --error=logs/train_8b_%j.err

# Load environment variables from .env file
set -a
source .env
set +a

# Activate virtual environment
source qlora_env/bin/activate

# Run the training script
python Fine-Tuning-QLoRA/train_llama8b.py

echo "Training completed!"
