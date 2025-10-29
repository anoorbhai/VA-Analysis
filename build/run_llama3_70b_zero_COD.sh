#!/bin/bash
#SBATCH --job-name=llama3
#SBATCH --output=llama3_70b_COD_cods%j.out
#SBATCH --error=llama3_70b_COD_cods%j.err
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1

# Change to script directory
cd /home/seedatr/VA-Analysis

ollama create Llama3_70b_zero_COD -f E2:COD_ICD10_List/Models/Llama3_70b_zero_COD

# Run the Python script
python E2:COD_ICD10_List/Prompting/Llama3_70b_ZeroShot_COD.py