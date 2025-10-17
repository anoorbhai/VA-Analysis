#!/bin/bash
#SBATCH --job-name=llama3
#SBATCH --output=logs/llama3_70b_cods%j.out
#SBATCH --error=logs/llama3_70b_cods%j.err
#SBATCH --time=7-00:00:00
#SBATCH --reservation=gpu
#SBATCH --nodelist=n22
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Change to script directory
cd /home/noorbhaia/VA-Analysis

# Run the Python script
python Prompting/FewShot/Llama3_70b_FewShot_61.py