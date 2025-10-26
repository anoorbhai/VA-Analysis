#!/bin/bash
#SBATCH --job-name=llama3.1
#SBATCH --output=llama3.1_70b_train_cods%j.out
#SBATCH --error=llama3.1_70b_train_cods%j.err
#SBATCH --time=7-00:00:00
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Change to script directory
cd /home/seedatr/VA-Analysis

# Run the Python script
python Fine-Tuning/Llama3.1_70b_train.py