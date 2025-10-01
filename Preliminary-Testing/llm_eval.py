import json
import time
import subprocess
from typing import Dict, List

def call_llm(model: str, prompt: str) -> str:
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def load_questions(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        data = json.load(f)
    return data['questions']


def ask_questions_and_save(model: str, questions: List[Dict], output_file: str):
    with open(output_file, 'w') as out:
        for q in questions:
            qid = q['id']
            prompt = (
                f"Q: {q['stem']}\nOptions:\n" +
                '\n'.join([f"{label}. {opt}" for label, opt in zip(['A','B','C','D'], q.get('options', []))]) +
                "\n\nPlease answer by starting your response with the single correct option letter (A, B, C, or D), followed by a space and then your explanation. For example: 'B. Because ...'.\nAnswer:"
            )
            start_time = time.time()
            llm_response = call_llm(model, prompt)
            end_time = time.time()
            response_time = end_time - start_time
            out.write(f"{qid} | {q['difficulty']}\n")
            out.write(f"Prompt: {prompt}\n")
            out.write(f"LLM Full Response:\n{llm_response}\n")
            out.write(f"Response Time: {response_time:.2f} seconds\n\n")

def main():
    questions = load_questions('niche_15_questions.json')
    models = [
        ('llama3:70b', 'llama3_70b_output.txt'),
        ('llama4:scout', 'llama4_scout_output.txt'),
        ('llama4:maverick', 'llama4_maverick_output.txt'),
    ]
    for model, outfile in models:
        print(f"\nPrompting {model}...")
        ask_questions_and_save(model, questions, outfile)
        print(f"  Output saved to: {outfile}")

if __name__ == '__main__':
    main()
