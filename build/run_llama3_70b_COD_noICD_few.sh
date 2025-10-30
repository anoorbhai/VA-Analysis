#!/bin/bash
#SBATCH --job-name=llama3_COD_few
#SBATCH --output=logs/llama3_70b_va_COD_noICD_few%j.out
#SBATCH --error=logs/llama3_70b_va_COD_noICD_few%j.err
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1

ollama create llama3_70b_few_COD_no_ICD10 -f E3:COD_No_ICD10_List/Models/Llama3_70b_Few_COD_no_ICD10

# Run the Python script
python E3:COD_No_ICD10_List/Prompting/Llama3_70b_FewShot_COD_no_ICD10.py