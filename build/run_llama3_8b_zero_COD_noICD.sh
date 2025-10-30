#!/bin/bash
#SBATCH --job-name=llama3
#SBATCH --output=llama3_8b_COD_noICDcods%j.out
#SBATCH --error=llama3_8b_COD_noICDcods%j.err
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1

# Change to script directory
cd /home/seedatr/VA-Analysis

ollama create Llama3_8b_zero_COD_no_ICD10 -f E3:COD_No_ICD10_List/Models/Llama3_8b_zero_COD_no_ICD10

# Run the Python script
python E3:COD_No_ICD10_List/Prompting/Llama3_8b_ZeroShot_COD_no_ICD10.py