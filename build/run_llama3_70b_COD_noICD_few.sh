#!/bin/bash
#SBATCH --job-name=llama3_COD_few
#SBATCH --output=logs/llama3_70b_va_COD_noICD_few%j.out
#SBATCH --error=logs/llama3_70b_va_COD_noICD_few%j.err
#SBATCH --time=7-00:00:00
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Change to script directory
cd /home/noorbhaia/VA-Analysis

# Run the Python script
python Prompting/FewShot/Llama3_70b_FewShot_COD_no_ICD10.py