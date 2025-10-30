#!/bin/bash
#SBATCH --job-name=llama4_fewshot
#SBATCH --output=llama4_fewshot_%j.out
#SBATCH --error=llama4_fewshot_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Change to script directory
cd /home/seedatr/VA-Analysis

# Run the Python script
python Prompting/Llama4_FewShot.py