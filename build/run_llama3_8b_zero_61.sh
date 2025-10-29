#!/bin/bash
#SBATCH --job-name=llama3
#SBATCH --output=llama3_8b_61_cods%j.out
#SBATCH --error=llama3_8b_61_cods%j.err
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1

# Change to script directory
cd /home/seedatr/VA-Analysis

ollama create Llama3_8b_zero_61 -f E4:61_WHO_Causes/Models/Llama3_8b_zero_61

# Run the Python script
python E4:61_WHO_Causes/Prompting/Llama3_8b_ZeroShot_61.py