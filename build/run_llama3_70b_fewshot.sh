#!/bin/bash
#SBATCH --job-name=llama3_70b_va
#SBATCH --output=logs/llama3_70b_va_%j.out
#SBATCH --error=logs/llama3_70b_va_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Change to script directory
cd /home/seedatr/VA-Analysis

# Run the Python script
python Prompting/Llama3_70b_FewShot.py