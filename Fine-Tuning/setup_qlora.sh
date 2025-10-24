#!/bin/bash
"""
Setup script for QLoRA fine-tuning environment
"""

echo "Setting up QLoRA fine-tuning environment for Verbal Autopsy..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv_qlora" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv_qlora
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv_qlora/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support first
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core requirements
echo "Installing QLoRA requirements..."
pip install -r requirements_qlora.txt

# Install flash-attention (optional but recommended)
echo "Installing flash-attention for faster training..."
pip install flash-attn --no-build-isolation || echo "Flash attention installation failed - continuing without it"

# Verify CUDA installation
echo "Verifying CUDA installation..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo "Setup complete! Activate environment with: source .venv_qlora/bin/activate"
echo "Run training with: python finetune_qlora.py"