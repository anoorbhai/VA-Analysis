#!/bin/bash
#SBATCH --job-name=llama3
#SBATCH --output=llama3_8b_COD_cods%j.out
#SBATCH --error=llama3_8b_COD_cods%j.err
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1

# Change to script directory
cd /home/seedatr/VA-Analysis

ollama create Llama3_8b_Few_COD -f E2:COD_ICD10_List/Models/Llama3_8b_Few_COD

# Run the Python script
python E2:COD_ICD10_List/Prompting/Llama3_8b_FewShot_COD.py