#!/bin/bash
#SBATCH --job-name=llama3_COD_few
#SBATCH --output=llama3_8b_va_COD_noICD_few%j.out
#SBATCH --error=llama3_8b_va_COD_noICD_few%j.err
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1

cd /home/seedatr/VA-Analysis

ollama create Llama3_8b_Few_COD_no_ICD10 -f E3:COD_No_ICD10_List/Models/Llama3_8b_Few_COD_no_ICD10

# Run the Python script
python E3:COD_No_ICD10_List/Prompting/Llama3_8b_FewShot_COD_no_ICD10.py