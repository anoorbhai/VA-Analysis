import subprocess

def call_llm(model: str, prompt: str) -> str:
    """Call the LLM using the ollama CLI and return the full response."""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8").strip()

def main():
    import csv
    from datetime import datetime
    model = 'llama3:latest'
    instructions = (
        "You are an experienced medical physician assisting with analysis of verbal autopsy.\n"
        "Using only the information provided, provide the top three most likely causes of death.\n"
        "If information is insufficient, return unclassified.\nThis data was collected from a field research centre in South Africa's rural northeast."
    )
    # Read the first row of Narrative.csv (after header)
    csv_path = '/dataA/madiva/va/VA/Narrative.csv'
    cod_csv_path = '/dataA/madiva/va/VA/InterVA_COD.csv'
    target_anon_id = 'CYOZP'

   # Extract narrative row
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        row = None
        for r in reader:
            if r['AnonId'] == target_anon_id:
                row = r
                break

    if row is None:
        print(f"AnonId {target_anon_id} not found in Narrative.csv.")
        return

    # Extract fields
    anon_id = row['AnonId']
    dob = row['Dob']
    dod = row['Dod']
    sex = row['Sex']
    interview_date = row['InterviewDate']
    narrative = row['Narrative'].strip()
    # Calculate age at death
    try:
        dob_dt = datetime.strptime(dob, '%Y-%m-%d')
        dod_dt = datetime.strptime(dod, '%Y-%m-%d')
        age_years = dod_dt.year - dob_dt.year - ((dod_dt.month, dod_dt.day) < (dob_dt.month, dob_dt.day))
    except Exception:
        age_years = 'unknown'
    # Format sex
    sex_str = 'female' if sex.upper() == 'F' else 'male'
    # Build prompt
    prompt = (
        f"{instructions}"
        f"The ID is {anon_id}\nThis person was born on {dob} and died on {dod}, making them {age_years} years old at the time of death.\n"
        f"They were a {sex_str}\n. This data was collected from an interview conducted on {interview_date}.\n"
        f"Here is the free text narrative provided by the family member:\n{narrative}\n\n"
    )
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
    output_path = f"llm_output_{timestamp}.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('PROMPT ' + model + ":\n")
        f.write(prompt + '\n\n')
        f.write('LLM RESPONSE:\n')
        f.write(llm_response + '\n')
        f.write(cod_text)

if __name__ == '__main__':
    main()

