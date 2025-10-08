#!/bin/bash
#SBATCH --job-name=llama4_va
#SBATCH --output=llama4_va_%j.out
#SBATCH --error=llama4_va_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Change to script directory
cd /home/noorbhaia/VA-Analysis

# Run the Python script
python Prompting/Llama4_ZeroShot.py