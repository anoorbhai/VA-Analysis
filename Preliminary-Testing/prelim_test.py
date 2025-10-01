import subprocess

def call_llm(model: str, prompt: str) -> str:
    """Call the LLM using the ollama CLI and return the full response."""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8").strip()

def build_prompt(csv_path: str, target_anon_id: str, instructions: str = "") -> str:
    import csv

    # Extract row
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        row = None
        for r in reader:
            if r['AnonId'] == target_anon_id:
                row = r
                break

    if row is None:
        print(f"AnonId {target_anon_id} not found in {csv_path}.")
        return ""

    # Build prompt dynamically from all columns
    prompt_parts = []
    
    # Add instructions if provided
    if instructions:
        prompt_parts.append(instructions)
    
    # Add each column and its value
    for column_name, value in row.items():
        if value and value.strip():  # Only include non-empty values
            # Clean up the value (strip whitespace, handle long text)
            cleaned_value = value.strip()
            prompt_parts.append(f"{column_name}: {cleaned_value}")
    
    # Join all parts with newlines
    prompt = "\n".join(prompt_parts) + "\n\n"
    
    return prompt

def main():
    import csv
    from datetime import datetime
    model = 'llama3:latest'
    instructions = (
        "You are an experienced medical physician assisting with analysis of verbal autopsy.\n"
        "Using only the information provided, provide the most likely causes of death.\n"
        "If information is insufficient, return unclassified.\nThis data was collected from a field research centre in South Africa's rural northeast.\n"
    )

    csv_path = '/spaces/25G05/VA_merged_cases.csv'
    cod_csv_path = '/dataA/madiva/va/VA/InterVA_COD.csv'
    target_anon_id = 'BKFJZ'

    # Build the prompt using the updated function
    prompt = build_prompt(csv_path, target_anon_id, instructions)
    
    if not prompt:  # If build_prompt returned empty string (AnonId not found)
        return

    print(f"Prompting {model}...")
    llm_response = call_llm(model, prompt)
    print("LLM Response printed to file")

    # Extract COD from interVA_COD.csv
    cod_row = None
    with open(cod_csv_path, newline='', encoding='utf-8') as codfile:
        cod_reader = csv.DictReader(codfile)
        for cr in cod_reader:
            if cr['AnonId'] == target_anon_id:
                cod_row = cr
                break

    if cod_row:
        cod_text = (
            f"INTERVA COD:\n"
            f"CAUSE1: {cod_row['CAUSE1']} (Likelihood: {cod_row['LIK1']})\n"
            f"CAUSE2: {cod_row['CAUSE2']} (Likelihood: {cod_row['LIK2']})\n"
            f"CAUSE3: {cod_row['CAUSE3']} (Likelihood: {cod_row['LIk3']})\n"
        )
    else:
        cod_text = "INTERVA COD: Not found for this AnonId.\n"

    # Write prompt and response to output file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"/spaces/25G05/llm_output_{timestamp}.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('PROMPT ' + model + ":\n")
        f.write(prompt + '\n\n')
        f.write('LLM RESPONSE:\n')
        f.write(llm_response + '\n')
        f.write(cod_text)

if __name__ == '__main__':
    main()

