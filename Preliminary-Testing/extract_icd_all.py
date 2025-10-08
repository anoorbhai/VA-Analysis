import csv

def extract_csv_to_text_simple(csv_filename, txt_filename):
    """
    Alternative version - simpler format output
    """
    try:
        with open(csv_filename, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            
            with open(txt_filename, 'w', encoding='utf-8') as txt_file:
                for row in csv_reader:
                    if len(row) >= 2:
                        # Simple format: Code - Description
                        txt_file.write(f"{row[0]} - {row[1]}\n")
                
        print(f"Successfully extracted CSV contents to {txt_filename}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_filename}'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Usage example
if __name__ == "__main__":
    csv_file = "Preliminary-Testing/ICD 10 Codes.csv"  # Replace with your CSV filename
    txt_file = "Preliminary-Testing/icd10_codes_all.txt"  # Output text filename
    
    extract_csv_to_text_simple(csv_file, txt_file)