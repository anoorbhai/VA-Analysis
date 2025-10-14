#!/bin/bash
#SBATCH --job-name=logs/llama3_8b_va
#SBATCH --output=logs/llama3_8b_va_%j.out
#SBATCH --error=logs/llama3_8b_va_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Change to script directory
cd /home/seedatr/VA-Analysis

# Run the Python script
python Prompting/FewShot/Llama3_8b_FewShot_COD_no_ICD10.py