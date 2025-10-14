#!/bin/bash
#SBATCH --job-name=llama3_COD
#SBATCH --output=logs/llama3_70b_va_COD_noICD%j.out
#SBATCH --error=logs/llama3_70b_va_COD_noICD%j.err
#SBATCH --time=7-00:00:00
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Change to script directory
cd /home/noorbhaia/VA-Analysis

# Run the Python script
python Prompting/ZeroShot/Llama3_70b_ZeroShot_COD_no_ICD10.py