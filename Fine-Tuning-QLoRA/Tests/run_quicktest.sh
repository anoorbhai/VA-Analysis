#!/bin/bash
#SBATCH --job-name=llama8b_qlora
#SBATCH --output=logs/llama8b_qlora_test%j.out
#SBATCH --error=logs/llama8b_qlora_test%j.err
#SBATCH --nodelist=n20
#SBATCH --gres=gpu:1
#SBATCH --reservation=gpu

# Load environment variables from .env file
set -a
source .env
set +a

# Activate virtual environment
source qlora_env/bin/activate

echo "=========================================="
echo "Running QUICK TEST of LLaMA-8B QLoRA"
echo "This will:"
echo "  - Use only 10 training samples"
echo "  - Use only 5 validation samples"
echo "  - Train for 1 epoch"
echo "  - Use reduced model parameters"
echo "=========================================="

# Run the quick test training script
python Fine-Tuning-QLoRA/train_llama8b_quicktest.py

echo ""
echo "=========================================="
echo "Quick test completed!"
echo "Check the output above for any errors"
echo "=========================================="
