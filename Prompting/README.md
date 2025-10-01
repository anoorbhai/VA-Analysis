# Llama3 Zero-Shot Verbal Autopsy Analysis

This script processes verbal autopsy data using Llama3 to predict causes of death based on symptom data and narratives.

## Setup

1. **Install Ollama**: Make sure Ollama is installed and running on your system
   ```bash
   # Start Ollama service
   ollama serve
   ```

2. **Build the VA Model**: Create the specialized verbal autopsy model from the Modelfile
   ```bash
   cd /home/noorbhaia/VA-Analysis/Prompting
   ollama create llama3_va -f Modelfile
   ```

3. **Install Python Dependencies**: Make sure you have the required Python packages
   ```bash
   pip install pandas requests
   ```

## Usage

### Basic Usage
```bash
cd /home/noorbhaia/VA-Analysis/Prompting
python Llama3_ZeroShot.py
```

### Test with Limited Cases
To test with only the first 5 cases, edit the script and uncomment the line:
```python
max_cases = 5  # Uncomment for testing with first 5 cases
```

## Input and Output

- **Input**: `/dataA/madiva/va/student/madiva_va_dataset_20250924.csv`
- **Output**: `/home/noorbhaia/VA-Analysis/Prompting/llama3_zeroshot_results.csv`

The script excludes the following fields as requested:
- cause1, prob1, cause2, prob2, cause3, prob3

## Output Format

The output CSV contains:
- `id`: Individual ID from the dataset
- `cause_of_death`: Predicted cause of death (short description)
- `icd10_code`: ICD-10 code for the predicted cause
- `confidence`: Confidence percentage (0-100)
- `time_taken_seconds`: Processing time for each case
- `processed_at`: Timestamp when the case was processed

## Features

- **Error Handling**: Robust error handling with logging
- **Intermediate Saves**: Saves progress every 10 cases to prevent data loss
- **Resume Capability**: Can be modified to resume from specific case index
- **Timing**: Measures and reports processing time per case
- **Logging**: Comprehensive logging to file and console

## Monitoring Progress

Check the log file for detailed progress:
```bash
tail -f llama3_zeroshot.log
```

## Expected Processing Time

Processing ~20,000 cases may take several hours depending on:
- Model size and complexity
- System resources (CPU/GPU)
- Network latency to Ollama API

## Troubleshooting

1. **Connection Issues**: Ensure Ollama is running (`ollama serve`)
2. **Model Not Found**: Build the model first (`ollama create llama3_va -f Modelfile`)
3. **Memory Issues**: Close other applications or reduce batch size
4. **Timeout Errors**: Increase timeout values in the script if needed

## File Structure

```
/home/noorbhaia/VA-Analysis/Prompting/
├── Modelfile                          # LLM configuration and prompts
├── Llama3_ZeroShot.py                # Main processing script
├── llama3_zeroshot_results.csv       # Final results (generated)
├── llama3_intermediate_*.csv         # Intermediate backups (generated)
├── llama3_zeroshot.log              # Processing log (generated)
└── README.md                        # This file
```