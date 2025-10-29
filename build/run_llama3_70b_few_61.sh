#!/bin/bash
#SBATCH --job-name=llama3_61
#SBATCH --output=logs/llama3_70b_61%j.out
#SBATCH --error=logs/llama3_70b_61%j.err
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1

ollama create Llama3_70b_Few_61 -f E4:61_WHO_Causes/Models/Llama3_70b_Few_61

# Run the Python script
python E4:61_WHO_Causes/Prompting/Llama3_70b_FewShot_61.py