#!/bin/bash
#SBATCH --job-name=llama3
#SBATCH --output=70b__zero_cod_%j.out
#SBATCH --error=70b_zero_cod_%j.err
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1

ollama create llama3_70b_zero_COD_no_ICD10 -f E3:COD_No_ICD10_List/Models/Llama3_70b_zero_COD_no_ICD10

# Run the Python script
python E3:COD_No_ICD10_List/Prompting/Llama3_70b_ZeroShot_COD_no_ICD10.py