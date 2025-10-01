#!/usr/bin/env python3
"""
Llama3 Zero-Shot Verbal Autopsy Analysis Script

This script processes the VA dataset using Llama3 for cause of death prediction.
It excludes cause/probability fields and uses the LLM to make predictions based on
symptom data and narratives.
"""

import pandas as pd
import requests
import json
import time
import re
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llama3_zeroshot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
INPUT_CSV_PATH = "/dataA/madiva/va/student/madiva_va_dataset_20250924.csv"
OUTPUT_CSV_PATH = "/spaces/25G05/ZeroShot/llama3_zeroshot_results.csv"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3_VA:latest" 

# Fields to exclude as specified
EXCLUDE_FIELDS = ['cause1', 'prob1', 'cause2', 'prob2', 'cause3', 'prob3']

class LlamaVAProcessor:
    """Main processor class for VA analysis using Llama3"""
    
    def __init__(self):
        self.session = requests.Session()
        self.results = []
        self.check_ollama_connection()
    
    def check_ollama_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            # Test connection to Ollama
            test_url = "http://localhost:11434/api/tags"
            response = self.session.get(test_url, timeout=10)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                if MODEL_NAME in model_names:
                    logger.info(f"✓ Ollama is running and model '{MODEL_NAME}' is available")
                else:
                    logger.warning(f"⚠ Model '{MODEL_NAME}' not found. Available models: {model_names}")
                    logger.warning(f"Please build the model from the Modelfile first:")
                    logger.warning(f"ollama create {MODEL_NAME} -f /home/noorbhaia/VA-Analysis/Prompting/Modelfile")
            else:
                logger.error(f"Failed to connect to Ollama: HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.error("Please ensure Ollama is running: ollama serve")
        
    def load_dataset(self) -> pd.DataFrame:
        """Load the VA dataset and exclude specified fields"""
        logger.info(f"Loading dataset from {INPUT_CSV_PATH}")
        
        try:
            df = pd.read_csv(INPUT_CSV_PATH)
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            
            # Remove excluded fields
            columns_to_keep = [col for col in df.columns if col not in EXCLUDE_FIELDS]
            df_filtered = df[columns_to_keep]
            
            excluded_count = len(df.columns) - len(df_filtered.columns)
            logger.info(f"Excluded {excluded_count} fields: {EXCLUDE_FIELDS}")
            logger.info(f"Dataset now has {len(df_filtered.columns)} columns")
            
            return df_filtered
            
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {INPUT_CSV_PATH}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    


    def format_case_for_llm(self, row: pd.Series) -> str:
        """Format a single case for LLM input using all columns and values."""
        prompt_parts = []
        individual_id = row.get('individual_id', 'Unknown')
        prompt_parts.append(f"Case ID: {individual_id}")

        prompt_parts.append("\nSYMPTOM DATA:")
        for column_name, value in row.items():
            if column_name in EXCLUDE_FIELDS:
                continue
            if pd.isna(value) or (isinstance(value, str) and value.strip() == '') or value == '-':
                continue
            cleaned_value = str(value).strip()
            prompt_parts.append(f"{column_name}: {cleaned_value}")

        narrative = row.get('narrative', '')
        if pd.isna(narrative) or str(narrative).strip() == '':
            narrative = "No narrative provided"
        prompt_parts.append("\nNARRATIVE:")
        prompt_parts.append(str(narrative).strip())

        prompt_parts.append("\nPlease analyze this verbal autopsy case and provide your diagnosis.")

        return "\n".join(prompt_parts)

    
    def query_llm(self, prompt: str) -> Tuple[Optional[str], Optional[str], Optional[int], float]:
        """
        Query the Ollama LLM and parse the response
        
        Returns:
            Tuple of (cause_short, icd10_code, confidence, execution_time)
        """
        start_time = time.time()
        
        try:
            # Prepare the request payload
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
            
            # Make the API request
            logger.debug("Sending request to Ollama API")
            response = self.session.post(
                OLLAMA_API_URL, 
                json=payload
            )
            
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                # Parse the structured response
                cause_short, icd10_code, confidence = self.parse_llm_response(response_text)
                
                logger.debug(f"LLM response parsed: cause={cause_short}, icd10={icd10_code}, conf={confidence}")
                return cause_short, icd10_code, confidence, execution_time
                
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None, None, None, execution_time
                
        except requests.exceptions.Timeout:
            execution_time = time.time() - start_time
            logger.error("Request timed out")
            return None, None, None, execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error querying LLM: {e}")
            return None, None, None, execution_time
    
    def parse_llm_response(self, response_text: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """
        Parse the structured LLM response to extract cause, ICD10, and confidence
        
        Expected format:
        { "ID": "DOBMC", "CAUSE_SHORT": "Acute Respiratory Tract Infection (Pneumonia)", "ICD10": "J18.0", "CONFIDENCE": "90" }
        """
        try:
            data = json.loads(response_text)
            cause_short = data.get("CAUSE_SHORT")
            icd10_code = data.get("ICD10")
            confidence = data.get("CONFIDENCE")
            if confidence is not None:
                try:
                    confidence = int(confidence)
                except ValueError:
                    confidence = None
            return cause_short, icd10_code, confidence
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Response text: {response_text}")
            return None, None, None
    
    def process_cases(self, df: pd.DataFrame, start_idx: int = 0, max_cases: Optional[int] = None) -> List[Dict]:
        """
        Process cases through the LLM with timing and error handling
        
        Args:
            df: DataFrame with VA cases
            start_idx: Starting index for processing (for resuming)
            max_cases: Maximum number of cases to process (for testing)
            
        Returns:
            List of result dictionaries
        """
        results = []
        total_cases = len(df) if max_cases is None else min(max_cases, len(df))
        end_idx = start_idx + total_cases if max_cases else len(df)
        
        logger.info(f"Processing {total_cases} cases (indices {start_idx} to {end_idx-1})")
        
        for idx in range(start_idx, min(end_idx, len(df))):
            row = df.iloc[idx]
            individual_id = row.get('individual_id', f'Unknown_{idx}')
            
            logger.info(f"Processing case {idx+1}/{len(df)}: {individual_id}")
            
            try:
                # Format case for LLM
                prompt = self.format_case_for_llm(row)
                
                # Query LLM
                cause_short, icd10_code, confidence, execution_time = self.query_llm(prompt)
                
                # Store result
                result = {
                    'id': individual_id,
                    'cause_of_death': cause_short or 'ERROR',
                    'icd10_code': icd10_code or 'ERROR',
                    'confidence': confidence,
                    'time_taken_seconds': round(execution_time, 2),
                    'processed_at': datetime.now().isoformat()
                }
                
                results.append(result)
                
                # Log progress
                if cause_short:
                    logger.info(f"Success: {individual_id} -> {cause_short} ({icd10_code}), {execution_time:.2f}s")
                else:
                    logger.warning(f"Failed to get valid response for {individual_id}")
                
                # Save intermediate results every 10 cases
                if (idx + 1) % 10 == 0:
                    self.save_intermediate_results(results, idx + 1)
                
            except Exception as e:
                logger.error(f"Unexpected error processing case {individual_id}: {e}")
                result = {
                    'id': individual_id,
                    'cause_of_death': 'PROCESSING_ERROR',
                    'icd10_code': 'ERROR',
                    'confidence': None,
                    'time_taken_seconds': 0,
                    'processed_at': datetime.now().isoformat()
                }
                results.append(result)
        
        return results
    
    def save_intermediate_results(self, results: List[Dict], processed_count: int):
        """Save intermediate results to prevent data loss"""
        intermediate_file = f"/spaces/25G05/ZeroShot/llama3_intermediate_{processed_count}.csv"
        
        try:
            # Ensure the output directory exists
            output_dir = Path(intermediate_file).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            df_results = pd.DataFrame(results)
            df_results.to_csv(intermediate_file, index=False)
            logger.info(f"Saved intermediate results ({processed_count} cases) to {intermediate_file}")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")
    
    def save_final_results(self, results: List[Dict]):
        """Save final results to CSV"""
        try:
            df_results = pd.DataFrame(results)
            
            # Ensure the output directory exists
            output_dir = Path(OUTPUT_CSV_PATH).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df_results.to_csv(OUTPUT_CSV_PATH, index=False)
            
            logger.info(f"Final results saved to {OUTPUT_CSV_PATH}")
            logger.info(f"Processed {len(results)} cases total")
            
            # Print summary statistics
            successful_cases = len([r for r in results if r['cause_of_death'] not in ['ERROR', 'PROCESSING_ERROR']])
            avg_time = sum([r['time_taken_seconds'] for r in results]) / len(results)
            
            logger.info(f"Success rate: {successful_cases}/{len(results)} ({successful_cases/len(results)*100:.1f}%)")
            logger.info(f"Average processing time: {avg_time:.2f} seconds per case")
            
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")
            raise

    def load_valid_ids(self, clinician_cod_path: str) -> set:
        df_cod = pd.read_csv(clinician_cod_path)
        return set(df_cod['individual_id'].dropna().astype(str))

def main():
    """Main execution function"""
    logger.info("Starting Llama3 Zero-Shot VA Analysis")
    logger.info(f"Input: {INPUT_CSV_PATH}")
    logger.info(f"Output: {OUTPUT_CSV_PATH}")
    logger.info(f"Model: {MODEL_NAME}")
    
    processor = LlamaVAProcessor()
    
    try:
        valid_ids = processor.load_valid_ids("/dataA/madiva/va/student/madiva_va_clinician_COD_20250926.csv")
        df = processor.load_dataset()
        df_filtered = df[df['individual_id'].astype(str).isin(valid_ids)]
        
        # For testing, you can limit the number of cases
        max_cases = 5  # Uncomment for testing with first 5 cases
        # max_cases = None  # Process all cases
        
        # Process cases through LLM
        logger.info("Starting LLM processing...")
        results = processor.process_cases(df_filtered, max_cases=max_cases)
        
        # Save results
        processor.save_final_results(results)
        
        logger.info("Processing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    
if __name__ == "__main__":
    main()