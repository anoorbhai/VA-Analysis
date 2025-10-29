#!/bin/bash
#SBATCH --job-name=llama3_COD
#SBATCH --output=logs/70b_few_COD%j.out
#SBATCH --error=logs/70b_few_COD%j.err
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1

ollama create Llama3_70b_Few_COD -f E2:COD_ICD10_List/Models/Llama3_70b_Few_COD

# Run the Python script
python E2:COD_ICD10_List/Prompting/Llama3_70b_FewShot_COD.py