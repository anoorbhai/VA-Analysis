#!/usr/bin/env python3
import os
import json
import torch
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get Hugging Face token from environment
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_HUB_TOKEN not found in environment variables. Please set it in .env file.")

login(token=hf_token)

# Configuration
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_ADAPTER_PATH = "Fine-Tuning-QLoRA/llama3-8b-qlora-va/checkpoint-500"
TEST_DATA_PATH = "/spaces/25G05/Fine-Tuning/test.jsonl"
OUTPUT_DIR = Path("/spaces/25G05/Fine-Tuning")
OUTPUT_CSV = OUTPUT_DIR / f"inference_results_{log_timestamp}.csv"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1
TOP_P = 0.9
NUM_TEST_CASES = None  # Set to None to run on all test cases, or specify a number for quick testing


OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def load_model_and_tokenizer():
    """Load the base model with LoRA adapter and tokenizer."""
    logger.info(f"Loading base model: {BASE_MODEL}")
    
    # Configure 4-bit quantization
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load LoRA adapter
    logger.info(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
    
    # Set to evaluation mode
    model.eval()
    
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer


def format_messages_for_inference(messages: List[Dict]) -> str:
    """Convert messages to the format used during training."""
    conversation = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            conversation += f"<|system|>\n{content}\n"
        elif role == "user":
            conversation += f"<|user|>\n{content}\n"
        elif role == "assistant":
            conversation += f"<|assistant|>\n"
            
    return conversation


def extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract and parse JSON from model response."""
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response)
    
    for match in matches:
        try:
            data = json.loads(match)
            if all(key in data for key in ["ID", "CAUSE_SHORT", "SCHEME_CODE", "CONFIDENCE"]):
                return data
        except json.JSONDecodeError:
            continue
    
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        logger.warning(f"Could not extract JSON from response: {response[:200]}")
        return None


def run_inference(model, tokenizer, test_data: List[Dict]) -> List[Dict]:
    results = []
    
    logger.info(f"Running inference on {len(test_data)} test cases")
    
    # Track total inference time
    total_inference_start = time.time()
    inference_times = []
    
    for idx, example in enumerate(tqdm(test_data, desc="Processing test cases")):
        try:
            case_start_time = time.time()
            
            messages = example["messages"]
            
            expected_answer = None
            for msg in messages:
                if msg["role"] == "assistant":
                    expected_answer = msg["content"]
                    break
            
            prompt = format_messages_for_inference(messages)
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            generation_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generation_time = time.time() - generation_start
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            response = generated_text[len(prompt):].strip()
            
            predicted_json = extract_json_from_response(response)
            expected_json = extract_json_from_response(expected_answer) if expected_answer else None
            
            case_time = time.time() - case_start_time
            inference_times.append(case_time)
            
            result = {
                "test_case_index": idx,
                "raw_response": response,
                "predicted_json": json.dumps(predicted_json) if predicted_json else None,
                "expected_json": json.dumps(expected_json) if expected_json else None,
                "predicted_id": predicted_json.get("ID") if predicted_json else None,
                "predicted_cause": predicted_json.get("CAUSE_SHORT") if predicted_json else None,
                "predicted_scheme_code": predicted_json.get("SCHEME_CODE") if predicted_json else None,
                "predicted_confidence": predicted_json.get("CONFIDENCE") if predicted_json else None,
                "expected_id": expected_json.get("ID") if expected_json else None,
                "expected_cause": expected_json.get("CAUSE_SHORT") if expected_json else None,
                "expected_scheme_code": expected_json.get("SCHEME_CODE") if expected_json else None,
                "expected_confidence": expected_json.get("CONFIDENCE") if expected_json else None,
                "generation_time_seconds": generation_time,
                "total_case_time_seconds": case_time,
            }
            
            results.append(result)
            
            if (idx + 1) % 50 == 0 or (idx + 1) == len(test_data):
                avg_time = sum(inference_times) / len(inference_times)
                logger.info(f"Processed {idx + 1}/{len(test_data)} cases | Avg time per case: {avg_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Error processing test case {idx}: {str(e)}")
            results.append({
                "test_case_index": idx,
                "error": str(e)
            })
    
    total_inference_time = time.time() - total_inference_start
    avg_time_per_case = sum(inference_times) / len(inference_times) if inference_times else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TIMING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total inference time: {total_inference_time:.2f} seconds ({total_inference_time/60:.2f} minutes)")
    logger.info(f"Average time per case: {avg_time_per_case:.2f} seconds")
    logger.info(f"Fastest case: {min(inference_times):.2f} seconds" if inference_times else "N/A")
    logger.info(f"Slowest case: {max(inference_times):.2f} seconds" if inference_times else "N/A")
    logger.info(f"{'='*60}\n")
    
    return results


def save_results(results: List[Dict], output_path: Path):
    """Save results to CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")
    
    # Filter out error cases for accuracy calculation
    valid_cases = df[df["predicted_json"].notna()].copy()
    
    valid_cases['scheme_code_match'] = (
        valid_cases["predicted_scheme_code"].astype(str) == 
        valid_cases["expected_scheme_code"].astype(str)
    )
    
    def normalize_cause(cause):
        """Normalize cause text for comparison."""
        if pd.isna(cause):
            return ""
        # Convert to string, lowercase, strip whitespace and quotes
        return str(cause).lower().strip().strip('"').strip("'")
    
    valid_cases['cause_short_match'] = (
        valid_cases["predicted_cause"].apply(normalize_cause) == 
        valid_cases["expected_cause"].apply(normalize_cause)
    )
    
    # Print summary statistics
    total_cases = len(results)
    successful_parses = df["predicted_json"].notna().sum()

    scheme_code_matches = valid_cases['scheme_code_match'].sum()
    cause_short_matches = valid_cases['cause_short_match'].sum()
    both_matches = (valid_cases['scheme_code_match'] & valid_cases['cause_short_match']).sum()
    
    parse_rate = (successful_parses/total_cases*100) if total_cases > 0 else 0
    scheme_accuracy = (scheme_code_matches/successful_parses*100) if successful_parses > 0 else 0
    cause_accuracy = (cause_short_matches/successful_parses*100) if successful_parses > 0 else 0
    both_accuracy = (both_matches/successful_parses*100) if successful_parses > 0 else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"INFERENCE SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total test cases: {total_cases}")
    logger.info(f"Successfully parsed JSON: {successful_parses}/{total_cases} ({parse_rate:.1f}%)")
    logger.info(f"")
    logger.info(f"EXACT MATCH ACCURACY (on successfully parsed cases):")
    logger.info(f"  Scheme Code matches:  {scheme_code_matches}/{successful_parses} ({scheme_accuracy:.1f}%)")
    logger.info(f"  Cause Short matches:  {cause_short_matches}/{successful_parses} ({cause_accuracy:.1f}%)")
    logger.info(f"  Both correct:         {both_matches}/{successful_parses} ({both_accuracy:.1f}%)")
    logger.info(f"{'='*60}\n")
    
    match_summary_path = output_path.parent / f"match_analysis_{output_path.stem}.csv"
    match_df = valid_cases[[
        'test_case_index', 
        'predicted_scheme_code', 'expected_scheme_code', 'scheme_code_match',
        'predicted_cause', 'expected_cause', 'cause_short_match',
        'predicted_confidence'
    ]].copy()
    match_df.to_csv(match_summary_path, index=False)
    logger.info(f"Match analysis saved to: {match_summary_path}")


def main():
    """Main inference pipeline."""
    logger.info("Starting inference pipeline")
    
    script_start_time = time.time()
    
    model_load_start = time.time()
    model, tokenizer = load_model_and_tokenizer()
    model_load_time = time.time() - model_load_start
    logger.info(f"Model loading time: {model_load_time:.2f} seconds ({model_load_time/60:.2f} minutes)")
    
    logger.info(f"Loading test data from: {TEST_DATA_PATH}")
    dataset = load_dataset("json", data_files={"test": TEST_DATA_PATH})
    test_data = dataset["test"]
    
    if NUM_TEST_CASES is not None:
        original_size = len(test_data)
        test_data = test_data.select(range(min(NUM_TEST_CASES, len(test_data))))
        logger.info(f"Limited test set from {original_size} to {len(test_data)} cases for quick testing")
    else:
        logger.info(f"Loaded {len(test_data)} test cases")
    
    results = run_inference(model, tokenizer, test_data)
    
    save_results(results, OUTPUT_CSV)
    
    total_script_time = time.time() - script_start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"TOTAL SCRIPT TIME: {total_script_time:.2f} seconds ({total_script_time/60:.2f} minutes)")
    logger.info(f"{'='*60}\n")
    
    logger.info("Inference pipeline completed successfully!")


if __name__ == "__main__":
    main()
