#!/usr/bin/env python3
"""
Inference script for fine-tuned QLoRA model
Tests the fine-tuned model on verbal autopsy cases
"""

import torch
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VAInference:
    """Inference class for Verbal Autopsy model"""
    
    def __init__(self, base_model_path: str, adapter_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        logger.info("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA adapter
        logger.info("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        
    def format_prompt(self, case_data: dict) -> str:
        """Format case data into the model's expected prompt format"""
        system_message = """You are an experienced medical physician assisting with verbal autopsy analysis.
The provided data was collected from a field research centre in South Africa's rural northeast.

We will provide, for each case:
- A list of binary questions about medical history, lifestyle and symptoms, where:
  y = yes, n = no
- A free-text narrative provided by a family member.

Your task is to:
1) Identify the single most likely cause of death, based solely on the provided information.
2) Provide the corresponding VA scheme code (e.g., 01.04, 12.09).
3) Estimate your confidence level in your answer as a percentage.

If the information is insufficient to determine a cause of death, explicitly state that in the output.

Return ONE JSON object with EXACTLY these keys:
{
  "ID": "<the case id>",
  "CAUSE_SHORT": "<short english wording for the code>",
  "SCHEME_CODE": "<VA scheme code>",
  "CONFIDENCE": "<percentage confidence>"
}
Do not add other keys or text."""
        
        user_message = f"""Case ID: {case_data['id']}

SYMPTOM DATA:
{case_data['symptoms']}
Narrative: {case_data['narrative']}

Please analyze this verbal autopsy case and provide your diagnosis."""
        
        # Format in Llama 3 chat format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    
    def predict(self, case_data: dict, max_new_tokens: int = 256) -> str:
        """Generate prediction for a single case"""
        prompt = self.format_prompt(case_data)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def batch_predict(self, cases: list, output_file: str = None) -> list:
        """Predict on multiple cases"""
        results = []
        
        for i, case in enumerate(cases):
            logger.info(f"Processing case {i+1}/{len(cases)}: {case['id']}")
            
            try:
                prediction = self.predict(case)
                result = {
                    "case_id": case['id'],
                    "prediction": prediction,
                    "ground_truth": case.get('ground_truth', None)
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing case {case['id']}: {e}")
                results.append({
                    "case_id": case['id'],
                    "prediction": "ERROR",
                    "error": str(e),
                    "ground_truth": case.get('ground_truth', None)
                })
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results

def load_test_cases(file_path: str) -> list:
    """Load test cases from JSONL file"""
    cases = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            messages = data['messages']
            
            # Extract case information
            user_message = None
            assistant_message = None
            
            for msg in messages:
                if msg['role'] == 'user':
                    user_message = msg['content']
                elif msg['role'] == 'assistant':
                    assistant_message = msg['content']
            
            if user_message and assistant_message:
                # Parse case ID
                case_id = user_message.split('Case ID: ')[1].split('\n')[0]
                
                # Extract symptom data and narrative
                content_parts = user_message.split('SYMPTOM DATA:\n')[1]
                if 'Narrative: ' in content_parts:
                    symptoms, narrative = content_parts.split('Narrative: ', 1)
                    narrative = narrative.split('\n\nPlease analyze')[0]
                else:
                    symptoms = content_parts
                    narrative = ""
                
                case = {
                    'id': case_id,
                    'symptoms': symptoms.strip(),
                    'narrative': narrative.strip(),
                    'ground_truth': assistant_message
                }
                cases.append(case)
    
    return cases

def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned VA model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--test_file", type=str, 
                        default="/home/noorbhaia/VA-Analysis/Fine-Tuning/test.jsonl",
                        help="Path to test file")
    parser.add_argument("--output_file", type=str, 
                        default="inference_results.json",
                        help="Output file for results")
    parser.add_argument("--num_cases", type=int, default=10,
                        help="Number of test cases to run")
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = VAInference(args.base_model, args.adapter_path)
    
    # Load test cases
    logger.info(f"Loading test cases from {args.test_file}")
    test_cases = load_test_cases(args.test_file)
    
    # Limit number of cases if specified
    if args.num_cases > 0:
        test_cases = test_cases[:args.num_cases]
    
    logger.info(f"Running inference on {len(test_cases)} cases")
    
    # Run inference
    results = inferencer.batch_predict(test_cases, args.output_file)
    
    # Print sample results
    logger.info("\nSample results:")
    for i, result in enumerate(results[:3]):
        logger.info(f"\nCase {i+1}:")
        logger.info(f"ID: {result['case_id']}")
        logger.info(f"Prediction: {result['prediction']}")
        if result.get('ground_truth'):
            logger.info(f"Ground Truth: {result['ground_truth']}")

if __name__ == "__main__":
    main()