#!/bin/bash
#SBATCH --job-name=infer_llama3
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/inference_8b_%j.out
#SBATCH --error=logs/inference_8b_%j.err

# Load environment variables from .env file
set -a
source .env
set +a

# Activate virtual environment
source qlora_env/bin/activate

# Run the inference script
python Fine-Tuning-QLoRA/inference_llama3.1_8b.py

echo "Inference completed!"
