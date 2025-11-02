# VA-Analysis: Verbal Autopsy Analysis with Large Language Models

A comprehensive research framework for automated Cause of Death (COD) determination from Verbal Autopsy (VA) data using Large Language Models (LLMs). This project explores multiple prompting strategies and fine-tuning approaches to classify causes of death according to various coding schemes including ICD-10 and WHO's 61 standardized causes.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Experimental Approaches](#experimental-approaches)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

Verbal Autopsy is a systematic method for ascertaining causes of death in populations lacking complete vital registration systems. This project leverages state-of-the-art Large Language Models to automate the analysis of VA interviews and predict the underlying cause of death.

### Key Features

- **Multiple Experimental Approaches**: Four distinct experimental setups (E1-E4) testing different COD coding schemes
- **Comprehensive Model Support**: Integration with Llama 3 (8B, 70B) Models
- **Flexible Prompting Strategies**: Both zero-shot and few-shot learning implementations
- **Fine-Tuning Pipeline**: QLoRA-based fine-tuning for domain adaptation
- **Automated Evaluation**: Built-in evaluation metrics comparing LLM predictions against clinician-assigned CODs
- **SLURM Integration**: HPC cluster support for large-scale model inference

## 📁 Project Structure

```
VA-Analysis/
├── E1:No_List/              # Experiment 1: No predefined COD list
├── E2:COD_ICD10_List/       # Experiment 2: COD with ICD-10 list
├── E3:COD_No_ICD10_List/    # Experiment 3: COD without ICD-10 codes
├── E4:61_WHO_Causes/        # Experiment 4: WHO 61 standardized causes
├── Fine-Tuning-QLoRA/       # QLoRA fine-tuning implementations
├── build/                   # SLURM batch scripts
├── logs/                    # Execution logs and error files
├── OpenDay-UI/              # User interface for demo purposes
└── Preliminary-Testing/     # Initial experiments and prototypes
```

### Experimental Approaches

#### E1: No Predefined List
- **Objective**: Open-ended COD prediction without constraints
- **Models**: Llama 3 (8B, 70B), Llama 4
- **Strategies**: Zero-shot and Few-shot prompting

#### E2: COD with ICD-10 List
- **Objective**: Predict COD from a predefined list with ICD-10 codes
- **Models**: Llama 3 (8B, 70B)
- **Strategies**: Zero-shot and Few-shot prompting
- **Output**: ICD-10 formatted causes

#### E3: COD without ICD-10 Codes
- **Objective**: Predict COD from a list without ICD-10 codes
- **Models**: Llama 3 (8B, 70B)
- **Strategies**: Zero-shot and Few-shot prompting
- **Focus**: Clinical terminology without standardized coding

#### E4: WHO 61 Causes
- **Objective**: Classify deaths using WHO's standardized 61-cause framework
- **Models**: Llama 3 (8B, 70B)
- **Strategies**: Zero-shot and Few-shot prompting
- **Mapping**: Clinician ICD-10 codes mapped to WHO scheme codes


### Model Deployment

- **Inference**: Ollama API integration
- **Fine-tuning**: QLoRA (Quantized Low-Rank Adaptation)
- **Quantization**: 4-bit quantization for efficient training

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (for training/inference)
- Ollama (for model serving)
- Access to SLURM cluster (optional, for batch processing)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/anoorbhai/VA-Analysis.git
cd VA-Analysis
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
# For fine-tuning
cd Fine-Tuning-QLoRA
pip install -r requirements.txt

# Core dependencies include:
# - transformers
# - torch
# - pandas
# - datasets
# - peft
# - bitsandbytes
# - huggingface_hub
```

4. **Configure environment variables**
```bash
# Create .env file in project root
echo "HUGGINGFACE_HUB_TOKEN=your_token_here" > .env
```

5. **Set up Ollama** (for inference)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull llama3:8b
ollama pull llama3:70b
```

## 💻 Usage

### Running Inference

#### Zero-Shot Prompting
```bash
# E1: No List
python E1:No_List/Prompting/Llama3_70b_ZeroShot.py

# E2: With ICD-10 List
python E2:COD_ICD10_List/Prompting/Llama3_70b_ZeroShot_COD.py

# E4: WHO 61 Causes
python E4:61_WHO_Causes/Prompting/Llama3_70b_ZeroShot_61.py
```

#### Few-Shot Prompting
```bash
# E1: No List
python E1:No_List/Prompting/Llama3_70b_FewShot.py

# E2: With ICD-10 List
python E2:COD_ICD10_List/Prompting/Llama3_70b_FewShot_COD.py

# E4: WHO 61 Causes
python E4:61_WHO_Causes/Prompting/Llama3_70b_FewShot_61.py
```

### Running on SLURM Cluster

```bash
# Submit job for 70B model with few-shot prompting
sbatch build/run_llama3_70b_fewshot.sh

# Submit job for 8B model with zero-shot prompting
sbatch build/run_llama3_8b_zero_61.sh

# Submit job for COD with ICD-10
sbatch build/run_llama3_70b_COD.sh
```

### Fine-Tuning with QLoRA

```bash
cd Fine-Tuning-QLoRA

# Prepare training data
python prepare_training_data.py

# Train Llama 3 8B model
python train_llama3_8b.py

# Run inference with fine-tuned model
python inference_llama3_8b.py
```

### Evaluation

```bash
# Evaluate E1 results
python E1:No_List/eval.py

# Evaluate E2 results (with ICD-10)
python E2:COD_ICD10_List/eval.py

# Evaluate E4 results (WHO 61 causes)
python E4:61_WHO_Causes/eval_61.py
```

## 📊 Datasets

### Input Data Structure

The project uses MADIVA VA dataset with the following components:

1. **VA Interview Data**:
   - Individual demographics (ID, sex, age, location)
   - Interview metadata (date, interviewer)
   - Structured VA questions (coded as binary/categorical variables)

2. **Clinician COD Data**:
   - Ground truth COD assignments by medical professionals
   - ICD-10 coded causes
   - Used for evaluation and validation

3. **Mapping Files**:
   - `61_codes.csv`: WHO 61 cause scheme mapping
   - `icd10_codes_all.txt`: Complete ICD-10 code reference
   - `clinician_to_scheme_mapping.csv`: Clinician COD to scheme code mapping

### Data Preprocessing

- **Field Mapping**: VA interview codes converted to human-readable questions
- **Data Filtering**: 
  - Removal of missing ICD-10 codes (configurable)
  - Exclusion of R99 (ill-defined) codes (configurable)
  - Filtering of uncertainty codes in WHO 61 scheme

## 📈 Evaluation

### Evaluation Metrics

All experiments include comprehensive evaluation comparing LLM predictions against clinician-assigned CODs:

#### For E1 & E2 (eval.py):
1. **ICD-10 Root Match**: Exact match on 3-character ICD-10 root codes
2. **First Letter of ICD-10 Code Match**: Match of first letter of ICD-10 code to see if general cause category was correct. 

#### For E3 (eval_noICD.py):
1. **Exact Code Number Match**: Exact match on whole number corresponding to the cause. 

#### For E4 and the Fine Tuned Model:

1. **Exact Code Match**: Perfect match between predicted and clinician-assigned WHO 61 cause codes
2. **Chapter Match**: Match on the broader scheme chapter category (first digit of scheme code)
3. **CSMF (Cause-Specific Mortality Fraction)**: Accuracy of predicted population-level cause distribution compared to true distribution
4. **CCC (Chance-Corrected Concordance)**: Measures how often the model’s predicted cause of death agrees with the true cause after accounting for the agreement that could have happened by random chance.
5. **Uncertainty Rate**: Percentage of cases where model assigns uncertainty code (99.00) instead of specific cause
6. **Uncertainty Precision**: Among uncertainty predictions, percentage where clinician also assigned uncertainty
7. **Top-5 Cause Recalls**: Top 5 Causes and their recalls
8. **Top-5 Macro Recall**: Average recall across all causes when considering top 5 predictions


### Evaluation Configuration

```python
# Example configuration from eval_61.py
REMOVE_MISSING_SCHEME = True
TOPK = 5
UNCERTAINTY_CODE = "99.00"
EXCLUDE_UNCERTAINTY = True
```

### Output Files

Evaluation scripts generate timestamped CSV files with:
- Individual predictions vs. ground truth
- Match indicators

## 📋 Results

Results are organized by experiment and stored in timestamped directories. 

Each result file includes:
- Raw LLM outputs
- Parsed predictions
- Evaluation metrics
- Execution logs

## 🛠️ Configuration

### Key Configuration Files

1. **Environment Variables** (`.env`)
```env
HUGGINGFACE_HUB_TOKEN=your_token_here
```

2. **Model Configuration** (in Python scripts)
```python
MODEL_NAME = "Llama3_70b_Few:latest"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MAX_INPUT_TOKENS = 2048
```

3. **Training Configuration** (`train_llama3_8b.py`)
```python
BATCH_SIZE_PER_DEV = 2
GR_ACCUM_STEPS = 8
LR = 2e-4
NUM_EPOCHS = 3
LORA_R = 16
LORA_ALPHA = 32
```