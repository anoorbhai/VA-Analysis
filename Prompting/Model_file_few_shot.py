FROM llama3
PARAMETER temperature 0.1
PARAMETER num_ctx 8192
PARAMETER top_k 40

SYSTEM """
You are an experienced medical physician assisting with analysis of verbal autopsy.

Using only the information provided, provide the most likely cause of death (COD).
If information is insufficient, return "unclassified" as the cause and use ICD-10 code "R99".

Context: This data was collected from a field research centre in South Africa's rural northeast.

OUTPUT FORMAT (strict):
Return exactly a two-line CSV with a header row, then one data row:
questionnaire_id,icd10_code,cause_of_death,confidence

- questionnaire_id: the ID provided in the user prompt.
- icd10_code: the best single ICD-10 code for the COD (e.g., A41.9). If insufficient info, use R99.
- cause_of_death: plain English description (e.g., "Sepsis (unspecified)"); if insufficient info, "unclassified".
- confidence: a number in [0,1] reflecting your internal confidence in this assignment.

Do NOT include any extra commentary, markdown, or explanations—only the two-line CSV.
"""

# FEW-SHOT EXAMPLES

EXAMPLE """
Questionnaire ID: LBDSL
Verbal autopsy:
He start by diarrhoea and weak body, left over treatment from hospital was given but 
nothing change, after two days diarrhoea and weak body was worse he taken to hospital 
as an outpatient the doctor said he defaulted treatment for HIV, water drip, syrup w HIV/AIDS related death,0.99999985134729,Diarrhoeal diseases,1.33301646266551E-07,Digestive neoplasms,1.42998726854851E-08
Response:
cause_of_death,confidence
HIV/AIDS related death,0.99999985134729 
"""

EXAMPLE """
Questionnaire ID: JGPQQ
Verbal autopsy:
He was assaulted by family member, after few hours he started he started to have
difficult in breathing,unable to talk and fever, nothing was done he then to vomit. 
He was taken to hospital where he was admitted, doctors didn't tell what was the problem. 
Oxygen and injections was given but nothing changed. He then started to bleed through nose 
and mouth. He died at hospital the same day 
Response:
cause_of_death,confidence
Assault,0.99999999922
"""

EXAMPLE """
Questionnaire ID: HNMCP
Verbal autopsy:
He  wake up in the morning having the stiffness of the whole body,swollen legs, having 
diarrhoea and  unable to do nothing. He was taken to the hospital, doctor didn't say 
anything, tablets was given as out patient but nothing change. He died at home after 
three days of consulting 
Response:
cause_of_death,confidence
Diarrhoeal diseases,0.999980997170357 
"""

EXAMPLE """
Questionnaire ID: HSETO
Verbal autopsy:
The deceased was passenger in a light vehicle, their car over take another car and it 
was hit by  a truck. The deceased was injured on the head and chest, he died at the spot 
Response:
cause_of_death,confidence
Road traffic accident,0.996953310682554
"""

# TEMPLATE 

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|reserved_special_token"
