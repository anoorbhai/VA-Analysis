
import pandas as pd
import requests
import json
import time
import re
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging with timestamped filename
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'llama3_70b_fewshot_61_{log_timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
INPUT_CSV_PATH = "/dataA/madiva/va/student/madiva_va_dataset_20250924.csv"
# Generate timestamped output filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV_PATH = f"/spaces/25G05/Aaliyah/FewShot/llama3_70b_fewshot_61_results_{timestamp}.csv"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "Llama3_70b_Few_61:latest" 

# Exclude InsilicoVA predictions
EXCLUDE_FIELDS = ['cause1', 'prob1', 'cause2', 'prob2', 'cause3', 'prob3']

# Field code to English meaning mapping
FIELD_MAPPINGS = {
    'individual_id': 'Individual ID',
    'hdss_name': 'HDSS Name',
    'interview_date': 'Interview Date',
    'interviewer_id': 'Interviewer ID',
    'dob': 'Date of Birth',
    'dod': 'Date of Death',
    'sex': 'Sex',
    'i004a': 'Did s(he) die during the wet season?',
    'i004b': 'Did s(he) die during the dry season?',
    'i019a': 'Was he male?',
    'i019b': 'Was she female?',
    'i022a': 'Was s(he) aged 65 years or more at death?',
    'i022b': 'Was s(he) aged 50 to 64 years at death?',
    'i022c': 'Was s(he) aged 15 to 49 years at death?',
    'i022d': 'Was s(he) aged 5-14 years at death?',
    'i022e': 'Was s(he) aged 1 to 4 years at death?',
    'i022f': 'Was s(he) aged 1 to 11 months at death?',
    'i022g': 'Was s(he) aged < 1 month (28 days) at death?',
    'i022h': 'Was s(he) a live baby who died within 24 hours of birth?',
    'i022i': 'Was s(he) a baby who died between 24 and 48 hours of birth?',
    'i022j': 'Was s(he) a baby who died more than 48 hours from birth, but within the first week?',
    'i022k': 'Was s(he) a baby who died after the first week, but within the first month?',
    'i022l': 'Was she a woman aged 12-19 years at death?',
    'i022m': 'Was she a woman aged 20-34 years at death?',
    'i022n': 'Was she a woman aged 35 to 49 years at death?',
    'i059o': 'Was she married at the time of death?',
    'i077o': 'Did (s)he suffer from any injury or accident that led to her/his death?',
    'i079o': 'Was (s)he injured in a road traffic accident?',
    'i082o': 'Was (s)he injured in a non-road transport accident?',
    'i083o': 'Was (s)he injured in a fall?',
    'i084o': 'Was (s)he poisoned in any way?',
    'i085o': 'Did (s)he die of drowning?',
    'i086o': 'Was (s)he injured by the bite or sting of a venomous animal?',
    'i087o': 'Was (s)he injured by an animal or insect (non-venomous)',
    'i089o': 'Was (s)he injured by burns or fire?',
    'i090o': 'Was (s)he subject to violence (suicide, homicide,abuse)?',
    'i091o': 'Was (s)he injured by a fire arm?',
    'i092o': 'Was (s)he stabbed, cut or pierced?',
    'i093o': 'Was (s)he strangled?',
    'i094o': 'Was (s)he injured by a blunt force?',
    'i095o': 'Was (s)he injured by a force of nature?',
    'i096o': 'Was (s)he electrocuted?',
    'i098o': 'Was the injury accidental?',
    'i099o': 'Was the injury or accident self-inflicted?',
    'i100o': 'Was the injury or accident intentionally inflicted by someone else?',
    'i104o': 'Did the baby ever cry?',
    'i105o': 'Did the baby cry immediately after birth, even if only a little bit?',
    'i106a': 'Was it more than 5 minutes after birth before the baby first cried?',
    'i107o': 'Did the baby stop being able to cry?',
    'i108a': 'Did the baby stop crying more than a day before (s)he died?',
    'i109o': 'Did the baby ever move?',
    'i110o': 'Did the baby ever breathe?',
    'i111o': 'Did the baby breathe immediately after birth, even a little?',
    'i112o': 'Did the baby have a breathing problem?',
    'i113o': 'Was the baby given assistance to breathe at birth?',
    'i114o': 'If the baby didn\'t show any sign of life, was it born dead?',
    'i115o': 'Were there any bruises or signs of injury on child\'s body after the birth?',
    'i116o': 'Was the baby\'s body soft, pulpy and discoloured with the skin peeling away?',
    'i120a': 'Did the final illness last less than 3 weeks?',
    'i120b': 'Did the final illness last at least 3 weeks?',
    'i123o': 'Did (s)he die suddenly?',
    'i125o': 'Was there any diagnosis by a health professional of tuberculosis?',
    'i127o': 'Was there any diagnosis by a health professional of HIV/AIDS?',
    'i128o': 'Did (s)he have a recent positive test by a health professional for malaria?',
    'i129o': 'Did (s)he have a recent negative test by a health professional for malaria?',
    'i130o': 'Was there any diagnosis by a health professional of dengue fever?',
    'i131o': 'Was there any diagnosis by a health professional of measles?',
    'i132o': 'Was there any diagnosis by a health professional of high blood pressure?',
    'i133o': 'Was there any diagnosis by a health professional of heart disease?',
    'i134o': 'Was there any diagnosis by a health professional of diabetes?',
    'i135o': 'Was there any diagnosis by a health professional of asthma?',
    'i136o': 'Was there any diagnosis by a health professional of epilepsy?',
    'i137o': 'Was there any diagnosis by a health professional of cancer?',
    'i138o': 'Was there any diagnosis by a health professional of Chronic Obstructive Pulmonary Disease (COPD)?',
    'i139o': 'Was there any diagnosis by a physician or health worker of dementia?',
    'i140o': 'Was there any diagnosis by a health professional of depression?',
    'i141o': 'Was there any diagnosis by a health professional of stroke?',
    'i142o': 'Was there any diagnosis by a health professional of sickle cell disease?',
    'i143o': 'Was there any diagnosis by a health professional of kidney disease?',
    'i144o': 'Was there any diagnosis by a health professional of liver disease?',
    'i147o': 'During the illness that led to death, did (s)he have a fever?',
    'i148a': 'Did the fever last less than a week before death?',
    'i148b': 'Did the fever last at least one week, but less than 2 weeks before death?',
    'i148c': 'Did the fever last at least 2 weeks before death?',
    'i149o': 'Did the fever continue until death?',
    'i150a': 'Was the fever severe?',
    'i151a': 'Was the fever continuous?',
    'i152o': 'Did (s)he have night sweats?',
    'i153o': 'During the illness that led to death, did (s)he have a cough?',
    'i154a': 'Did the cough last less than 3 weeks before death?',
    'i154b': 'Did the cough last at least 3 weeks before death?',
    'i155o': 'Was the cough productive, with sputum?',
    'i156o': 'Was the cough very severe?',
    'i157o': 'Did (s)he cough up blood?',
    'i158o': 'Did (s)he make a whooping sound when coughing?',
    'i159o': 'During the illness that led to death, did (s)he have any difficulty breathing?',
    'i161a': 'Did the difficult breathing last for at least 3 days before death?',
    'i165a': 'Was the difficult breathing continuous during this period?',
    'i166o': 'Did (s)he have fast breathing?',
    'i167a': 'Did the fast breathing last for less than two weeks before death?',
    'i167b': 'Did the fast breathing last for at least 2 weeks before death?',
    'i168o': 'Did (s)he have breathlessness?',
    'i169a': 'Did the breathlessness last for less than 2 weeks before death?',
    'i169b': 'Did the breathlessness last for at least 2 weeks before death?',
    'i170o': 'Was (s)he unable to carry out daily routines due to breathlessness?',
    'i171o': 'Was (s)he breathless while lying flat?',
    'i172o': 'Did you see the lower chest wall/ribs being pulled in as the child breathed?',
    'i173a': 'Did his/her breathing sound like wheezing or grunting?',
    'i174o': 'During the illness that led to death, did (s)he have chest pain?',
    'i175o': 'Was the chest pain severe?',
    'i176a': 'Did (s)he experience chest pain at least 3 days before death?',
    'i178a': 'Did the chest pain last for at least 30 minutes?',
    'i181o': 'Did (s)he have diarrhoea?',
    'i182a': 'Did (s)he have diarrhoea for less than 2 weeks before death?',
    'i182b': 'Did (s)he have diarrhoea for at least 2 weeks but less than 4 weeks before death?',
    'i182c': 'Did (s)he have diarrhoea for at least 4 weeks before death?',
    'i183a': 'Did the baby or child have at least 4 stools on the day that loose liquid stools were most frequent?',
    'i184a': 'Did the frequent loose or liquid stools start at least 3 days before death?',
    'i185o': 'Did the frequent loose or liquid stools continue up until death?',
    'i186o': 'At any time during the final illness was there blood in the stools?',
    'i187o': 'Was there blood in the stool up until death?',
    'i188o': 'During the illness that led to death, did (s)he vomit?',
    'i189o': 'Did (s)he vomit in the week preceding the death?',
    'i190o': 'Did (s)he vomit for at least 3 days before death?',
    'i191o': 'Was there blood in the vomit?',
    'i192o': 'Was the vomit black?',
    'i193o': 'During the illness that led to death, did (s)he have any abdominal (belly) problem?',
    'i194o': 'Did (s)he have abdominal pain?',
    'i195o': 'Was the abdominal pain severe?',
    'i197a': 'Did (s)he have severe abdominal pain for less than 2 weeks before death?',
    'i197b': 'Did (s)he have severe abdominal pain for at least 2 weeks before death?',
    'i199a': 'Was the pain in the upper abdomen?',
    'i199b': 'Was the pain in the lower abdomen?',
    'i200o': 'Did (s)he have a more than usually protruding abdomen?',
    'i201a': 'Did (s)he have a more than usually protruding abdomen for less than 2 weeks before death?',
    'i201b': 'Did (s)he have a more than usually protruding abdomen for at least 2 weeks before death?',
    'i203a': 'Did (s)he develop the protruding abdomen rapidly?',
    'i204o': 'Did (s)he have any mass in the abdomen?',
    'i205a': 'Did (s)he have a mass in the abdomen for less than 2 weeks before death?',
    'i205b': 'Did (s)he have a mass in the abdomen for at least 2 weeks before death?',
    'i207o': 'During the illness that led to death, did (s)he have a severe headache?',
    'i208o': 'During the illness that led to death, did (s)he have a stiff neck?',
    'i209a': 'Did (s)he have a stiff neck for less than one week before death?',
    'i209b': 'Did (s)he have a stiff neck for at least one week before death?',
    'i210o': 'During the illness that led to death, did (s)he have a painful neck?',
    'i211a': 'Did (s)he have a painful neck for at least one week before death?',
    'i212o': 'During the illness that led to death, did (s)he have mental confusion?',
    'i213o': 'Did (s)he have mental confusion for at least 3 months before death?',
    'i214o': 'During the illness that led to death, was (s)he unconscious?',
    'i215o': 'Was (s)he unconscious for at least 24 hours before death?',
    'i216a': 'Was (s)he unsconscious for at least 6 hours before death?',
    'i217o': 'Did the unconsciousness start suddenly, quickly (at least within a single day)?',
    'i218o': 'Did the unconsciousness continue until death?',
    'i219o': 'During the illness that led to death, did (s)he have any convulsions?',
    'i220o': 'Did (s)he experience any generalized convulsions or fits?',
    'i221a': 'Did the convulsions last for less than 10 minutes?',
    'i221b': 'Did the convulsions last for at least 10 minutes?',
    'i222o': 'Did (s)he become unconscious immediately after the convulsion?',
    'i223o': 'During the illness that led to death, did (s)he have any urine problems?',
    'i224o': 'Did (s)he stop urinating?',
    'i225o': 'Did (s)he go to urinate more often than usual?',
    'i226o': 'Did (s)he pass blood in the urine?',
    'i227o': 'During the illness that led to death, did (s)he have any sores or ulcers anywhere?',
    'i228o': 'Did (s)he have sores?',
    'i229o': 'Did the sores have clear fluid and/or pus?',
    'i230o': 'Did (s)he have an ulcer (pit) on the foot?',
    'i231o': 'Did any ulcer ooze pus?',
    'i232a': 'Did the ulcer ooze pus for at least 2 weeks?',
    'i233o': 'During the illness that led to death, did (s)he have any skin rash?',
    'i234a': 'Did (s)he have the skin rash for less than one week?',
    'i234b': 'Did (s)he have the skin rash for at least one week?',
    'i235a': 'Did (s)he have a rash on the face?',
    'i235b': 'Did (s)he have a rash on the trunk or abdomen?',
    'i235c': 'Did (s)he have a rash on the extremities?',
    'i235d': 'Did (s)he have a rash everywhere?',
    'i236o': 'Did (s)he have measles rash?',
    'i237o': 'Did (s)he ever have shingles or herpes zoster?',
    'i238o': 'During the illness that led to death, did her/his skin flake off in patches?',
    'i239o': 'During the illness that led to death, did he/she have areas of the skin that turned black?',
    'i240o': 'During the illness that led to death, did he/she have areas of the skin with redness and swelling?',
    'i241o': 'During the illness that led to death, did (s)he bleed from anywhere?',
    'i242o': 'Did (s)he bleed from the nose, mouth or anus?',
    'i243o': 'During the illness that led to death, did (s)he have noticeable weight loss?',
    'i244o': 'Was (s)he severely thin or wasted?',
    'i245o': 'During the illness that led to death, did s/he have a whitish rash inside the mouth or on the tongue?',
    'i246o': 'During the illness that led to death, did (s)he have stiffness of the whole body or was unable to open the mouth?',
    'i247o': 'During the illness that led to death, did (s)he have puffiness of the face?',
    'i248a': 'Did (s)he have puffiness of the face for at least one week before death?',
    'i249o': 'During the illness that led to death, did (s)he have swollen legs or feet?',
    'i250a': 'Did the swelling last for at least 3 days before death?',
    'i251o': 'Did (s)he have both feet swollen?',
    'i252o': 'During the illness that led to death, did (s)he have general puffiness all over his/her body?',
    'i253o': 'During the illness that led to death, did (s)he have any lumps?',
    'i254o': 'Did (s)he have any lumps or lesions in the mouth?',
    'i255o': 'Did (s)he have any lumps in the neck?',
    'i256o': 'Did (s)he have any lumps in the armpit?',
    'i257o': 'Did (s)he have any lumps in the groin?',
    'i258o': 'During the illness that led to death, was (s)he in any way paralysed?',
    'i259o': 'Did (s)he have paralysis of only one side of the body?',
    'i260a': 'Was only the right side of the body paralysed?',
    'i260b': 'Was only the left side of the body paralysed?',
    'i260c': 'Was only the lower part of the body paralysed?',
    'i260d': 'Was only the upper part of the body paralysed?',
    'i260e': 'Was only one leg paralysed?',
    'i260f': 'Was only one arm paralysed?',
    'i260g': 'Was the entire body paralysed?',
    'i261o': 'During the illness that led to death, did (s)he have difficulty swallowing?',
    'i262a': 'Did (s)he have difficulty swallowing for at least one week before death?',
    'i263a': 'Did (s)he have difficulty with swallowing solids?',
    'i263b': 'Did (s)he have difficulty with swallowing liquids?',
    'i264o': 'Did (s)he have pain upon swallowing?',
    'i265o': 'During the illness that led to death, did (s)he have yellow discolouration of the eyes?',
    'i266a': 'Did (s)he have the yellow discolouration for at least 3 weeks before death?',
    'i267o': 'During the illness that led to death, did her/his hair change to a reddish or yellowish colour?',
    'i268o': 'During the illness that led to death, did (s)he look pale (thinning/lack of blood) or have pale palms, eyes or nail beds?',
    'i269o': 'During the illness that led to death, did (s)he have sunken eyes?',
    'i270o': 'During the illness that led to death, did (s)he drink a lot more water than usual?',
    'i271o': 'Was the baby able to suckle or bottle-feed within the first 24 hours after birth?',
    'i272o': 'Did the baby ever suckle in a normal way?',
    'i273o': 'Did the baby stop suckling?',
    'i274a': 'Did the baby stop suckling on the 2nd day of life or later?',
    'i275o': 'Did the baby have convulsions starting within the first 24 hours of life?',
    'i276o': 'Did the baby have convulsions starting more than 24 hours after birth?',
    'i277o': 'Did the baby\'s body become stiff, with the back arched backwards?',
    'i278o': 'During the illness that led to death, did the baby have a bulging or raised fontanelle?',
    'i279o': 'During the illness that led to death, did the baby have a sunken fontanelle?',
    'i281o': 'During the illness that led to death, did the baby become unresponsive or unconscious?',
    'i282o': 'Did the baby become unresponsive or unconscious soon after birth, within less than 24 hours?',
    'i283o': 'Did the baby become unresponsive or unconscious more than 24 hours after birth?',
    'i284o': 'During the illness that led to death, did the baby become cold to touch?',
    'i285a': 'Was the baby more than 3 days old when it started feeling cold to touch?',
    'i286o': 'During the illness that led to death, did the baby become lethargic, after a period of normal activity?',
    'i287o': 'Did the baby have redness or discharge from the umbilical cord stump?',
    'i288o': 'During the illness that led to death, did the baby have skin ulcer(s) or pits?',
    'i289o': 'During the illness that led to death, did the baby have yellow skin, palms (hand) or soles (foot)?',
    'i290o': 'Did the baby or infant appear to be healthy and then just die suddenly?',
    'i294o': 'During the illness that led to death, did she have any swelling or lump in the breast?',
    'i295o': 'During the illness that led to death, did she have any ulcers (pits) in the breast?',
    'i296o': 'Did she ever have a period or menstruate?',
    'i297o': 'During the illness that led to death, did she have excessive vaginal bleeding in between menstrual periods?',
    'i298o': 'Was the bleeding excessive?',
    'i299o': 'Did her menstrual period stop naturally because of menopause?',
    'i300o': 'Did she have vaginal bleeding after cessation of menstruation?',
    'i301o': 'Was there excessive vaginal bleeding in the week prior to death?',
    'i302o': 'At the time of death was her period overdue?',
    'i303a': 'Had her period been overdue for at least 4 weeks?',
    'i304o': 'Did she have a sharp pain in her abdomen shortly before death?',
    'i305o': 'Was she pregnant at the time of death?',
    'i306o': 'Did she die within 6 weeks of delivery, abortion or miscarriage?',
    'i309o': 'Was she, or had she been, pregnant for less than 6 months when she died?',
    'i310o': 'Please confirm: When she died, she was NEITHER pregnant NOR had recently been pregnant NOR had recently delivered when she died - is that right?',
    'i312o': 'Did she die during labour, but before delivery?',
    'i313o': 'Did she die after delivering a baby?',
    'i314o': 'Did she die within 24 hours after delivery?',
    'i315o': 'Did she die within 6 weeks of childbirth?',
    'i316o': 'Did she give birth to a live baby (within 6 weeks of her death)?',
    'i317o': 'Did she die during or after a multiple pregnancy?',
    'i318o': 'Was she breastfeeding the child in the days before death?',
    'i319a': 'Did she die during or after her first pregnancy?',
    'i319b': 'Did she have four or more pregnancies before this one?',
    'i320o': 'Had she had any previous Caesarean section?',
    'i321o': 'During pregnancy, did she suffer from high blood pressure?',
    'i322o': 'Did she have foul smelling vaginal discharge during pregnancy or after delivery?',
    'i323o': 'During the last 3 months of pregnancy, did she suffer from convulsions?',
    'i324o': 'During the last 3 months of pregnancy did she suffer from blurred vision?',
    'i325o': 'Did she have excessive bleeding during pregnancy or shortly after delivery?',
    'i326o': 'Was there vaginal bleeding during the first 6 months of pregnancy?',
    'i327o': 'Was there vaginal bleeding during the last 3 months of pregnancy but before labour started?',
    'i328o': 'Did she have excessive bleeding during labour, before delivery?',
    'i329o': 'Did she have excessive bleeding after delivery or abortion?',
    'i330o': 'Was the placenta completely delivered?',
    'i331o': 'Did she deliver or try to deliver an abnormally positioned baby?',
    'i332a': 'Did her labour last longer than 24 hours?',
    'i333o': 'Did she attempt to terminate the pregnancy?',
    'i334o': 'Did she recently have a pregnancy that ended in an abortion (spontaneous or induced)?',
    'i335o': 'Did she die during an abortion?',
    'i336o': 'Did she die within 6 weeks of having an abortion?',
    'i337a': 'Did the mother deliver at a health facility or clinic?',
    'i337b': 'Did the mother deliver at home?',
    'i337c': 'Did the mother deliver elsewhere (not at a health facility nor at home)?',
    'i338o': 'Did she receive professional assistance during the delivery?',
    'i340o': 'Did she have an operation to remove her uterus shortly before death?',
    'i342o': 'Was the delivery normal vaginal, without forceps or vacuum?',
    'i343o': 'Was the delivery vaginal, with forceps or vacuum?',
    'i344o': 'Was the delivery a Caesarean section?',
    'i347o': 'Was her baby born more than one month early?',
    'i354o': 'Was the child part of a multiple birth?',
    'i355a': 'If the child was part of a multiple birth, was it born first?',
    'i356o': 'Is the child\'s mother still alive?',
    'i357o': 'Did the child\'s mother die during or shortly after the delivery?',
    'i358a': 'Did the child\'s mother die in the baby\'s first year of life?',
    'i360a': 'Was the baby born in a health facility or clinic?',
    'i360b': 'Was the baby born at home?',
    'i360c': 'Was the baby born somewhere else (e.g. on the way to a clinic)?',
    'i361o': 'Did the mother receive professional assistance during the delivery?',
    'i362o': 'At birth, was the baby of usual size?',
    'i363o': 'At birth, was the baby smaller than normal (weighing under 2.5 kg)?',
    'i364o': 'At birth, was the baby very much smaller than usual, (weighing under 1 kg)?',
    'i365o': 'At birth, was the baby larger than normal (weighing over 4.5 kg)?',
    'i367a': 'Was the baby born during the ninth month (at least 37 weeks) of pregnancy?',
    'i367b': 'Was the baby born during the eighth month (34 to 37 weeks) of pregnancy?',
    'i367c': 'Was the baby born before the eighth month (less than 34 weeks) of pregnancy?',
    'i368o': 'Were there any complications in the late part of the pregnancy (defined as the last 3 months), but before labour?',
    'i369o': 'Were there any complications during labour or delivery?',
    'i370o': 'Was any part of the baby physically abnormal at time of delivery? (for example: body part too large or too small, additional growth on body)?',
    'i371o': 'Did the baby/child have a swelling or defect on the back?',
    'i372o': 'Did the baby/child have a very large head?',
    'i373o': 'Did the baby/child have a very small head?',
    'i376o': 'Was the baby moving in the last few days before the birth?',
    'i377o': 'Did the baby stop moving in the womb before labour started?',
    'i382a': 'Did labour and delivery take more than 24 hours?',
    'i383o': 'Was the baby born 24 hours or more after the waters broke?',
    'i384o': 'Was the liquor foul smelling when the waters broke?',
    'i385a': 'Was the liquor a green or brown colour when the waters broke?',
    'i387o': 'Was the delivery normal vaginal, without forceps or vacuum?',
    'i388o': 'Was the delivery vaginal, with forceps or vacuum?',
    'i389o': 'Was the delivery a Caesarean section?',
    'i391o': 'Did the child\'s mother receive any vaccinations since reaching adulthood including during this pregnancy?',
    'i393o': 'Did the mother receive tetanus toxoid (TT) vaccine?',
    'i394a': 'Was this baby born from the mother\'s first pregnancy?',
    'i394b': 'Did the baby\'s mother have four or more births before this one?',
    'i395o': 'During labour, did the baby\'s mother suffer from fever?',
    'i396o': 'During the last 3 months of pregnancy, labour or delivery, did the baby\'s mother suffer from high blood pressure?',
    'i397o': 'Did the baby\'s mother have diabetes mellitus?',
    'i398o': 'Did the baby\'s mother have foul smelling vaginal discharge during pregnancy or after delivery?',
    'i399o': 'During the last 3 months of pregnancy, labour or delivery, did the baby\'s mother suffer from convulsions?',
    'i400o': 'During the last 3 months of pregnancy did the baby\'s mother suffer from blurred vision?',
    'i401o': 'Did the baby\'s mother have severe anaemia?',
    'i402o': 'Did the baby\'s mother have vaginal bleeding during the last 3 months of pregnancy but before labour started?',
    'i403o': 'Did the baby\'s bottom, feet, arm or hand come out of the vagina before its head?',
    'i404o': 'Was the umbilical cord wrapped more than once around the baby\'s neck at birth?',
    'i405o': 'Was the umbilical cord delivered first?',
    'i406o': 'Was the baby blue in colour at birth?',
    'i408o': 'Before the illness that led to death, was the baby/child growing normally?',
    'i411o': 'Did (s)he drink alcohol?',
    'i412o': 'Did (s)he use tobacco?',
    'i413o': 'Did (s)he smoke tobacco (cigarette, cigar, pipe, etc.)?',
    'i414a': 'Did (s)he use non-smoking tobacco?',
    'i415a': 'Did (s)he smoke at least 10 cigarettes daily?',
    'i418o': 'Did (s)he receive any treatment for the illness that led to death?',
    'i419o': 'Did (s)he receive oral rehydration salts?',
    'i420o': 'Did (s)he receive (or need) intravenous fluids (drip) treatment?',
    'i421o': 'Did (s)he receive (or need) a blood transfusion?',
    'i422o': 'Did (s)he receive (or need) treatment/food through a tube passed through the nose?',
    'i423o': 'Did (s)he receive (or need) injectable antibiotics?',
    'i424o': 'Did (s)he receive (or need) antiretroviral therapy (ART)?',
    'i425o': 'Did (s)he have (or need) an operation for the illness?',
    'i426o': 'Did (s)he have the operation within 1 month before death?',
    'i427o': 'Was (s)he discharged from hospital very ill?',
    'i428o': 'Did (s)he receive appropriate immunizations?',
    'i450o': 'In the final days before death, did s/he travel to a hospital or health facility?',
    'i451o': 'Did (s)he use motorised transport to get to the hospital or health facility?',
    'i452o': 'Were there any problems during admission to the hospital or health facility?',
    'i453o': 'Were there any problems with the way (s)he was treated (medical treatment, procedures, interpersonal attitudes, respect, dignity) in the hospital or health facility?',
    'i454o': 'Were there any problems getting medications, or diagnostic tests in the hospital or health facility?',
    'i455o': 'Does it take more than 2 hours to get to the nearest hospital or health facility from the deceased\'s household?',
    'i456o': 'In the final days before death, were there any doubts about whether medical care was needed?',
    'i457o': 'In the final days before death, was traditional medicine used?',
    'i458o': 'In the final days before death, did anyone use a telephone or cell phone to call for help?',
    'i459o': 'Over the course of illness, did the total costs of care and treatment prohibit other household payments?',
    'narrative': 'Narrative'
}

class LlamaVAProcessor:
    
    
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
                    logger.info(f"Ollama is running and model '{MODEL_NAME}' is available")
                else:
                    logger.warning(f"Model '{MODEL_NAME}' not found. Available models: {model_names}")
            else:
                logger.error(f"Failed to connect to Ollama: HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.error("Please ensure Ollama is running: ollama serve")
        
    def load_dataset(self) -> pd.DataFrame:

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
        """Format a single case for LLM input using human-readable field names and filtering out '-' values."""
        prompt_parts = []
        individual_id = row.get('individual_id', 'Unknown')
        prompt_parts.append(f"Case ID: {individual_id}")
        prompt_parts.append("\nSYMPTOM DATA:")
        
        for column_name, value in row.items():
            if column_name in EXCLUDE_FIELDS:
                continue
            if pd.isna(value) or (isinstance(value, str) and value == ''):
                continue
            
            # Skip fields with '-' values - only include 'y' and 'n' responses
            if isinstance(value, str) and value.strip() == '-':
                continue
            
            # For binary fields, only include if value is 'y' or 'n'
            if column_name.startswith('i') and isinstance(value, str):
                if value.strip().lower() not in ['y', 'n']:
                    continue
                    
            # Get human-readable field name
            field_name = FIELD_MAPPINGS.get(column_name, column_name)
            prompt_parts.append(f"{field_name}: {value}")
        
        narrative = row.get('narrative', '')
        if pd.isna(narrative) or str(narrative).strip() == '':
            narrative = "No narrative provided"
        
        
        prompt_parts.append("\nPlease analyze this verbal autopsy case and provide your diagnosis.")
        
        return "\n".join(prompt_parts)
    
    def query_llm(self, prompt: str) -> Tuple[Optional[str], Optional[str], Optional[int], float]:

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
                cause_short, code, confidence = self.parse_llm_response(response_text)
                
                logger.debug(f"LLM response parsed: cause={cause_short}, code={code}, conf={confidence}")
                return cause_short, code, confidence, execution_time
                
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
        
        try:
            data = json.loads(response_text)
            cause_short = data.get("CAUSE_SHORT")
            code = data.get("CODE")
            confidence = data.get("CONFIDENCE")
            if confidence is not None:
                try:
                    confidence = int(confidence)
                except ValueError:
                    confidence = None
            return cause_short, code, confidence
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Response text: {response_text}")
            return None, None, None
    
    def process_cases(self, df: pd.DataFrame, start_idx: int = 0, max_cases: Optional[int] = None) -> List[Dict]:
        
        results = []
        total_cases = len(df) if max_cases is None else min(max_cases, len(df))
        end_idx = start_idx + total_cases if max_cases else len(df)
        
        logger.info(f"Processing {total_cases} cases (indices {start_idx} to {end_idx-1})")
        
        # Track total processing time
        total_start_time = time.time()
        
        for idx in range(start_idx, min(end_idx, len(df))):
            row = df.iloc[idx]
            individual_id = row.get('individual_id', f'Unknown_{idx}')
            
            logger.info(f"Processing case {idx+1}/{len(df)}: {individual_id}")
            
            try:
                # Format case for LLM
                prompt = self.format_case_for_llm(row)
                
                # Query LLM
                cause_short, code, confidence, execution_time = self.query_llm(prompt)
                
                # Store result
                result = {
                    'id': individual_id,
                    'cause_of_death': cause_short or 'ERROR',
                    'code': code or 'ERROR',
                    'confidence': confidence,
                    'time_taken_seconds': round(execution_time, 2),
                    'processed_at': datetime.now().isoformat()
                }
                
                results.append(result)
                
                # Log progress
                if cause_short:
                    logger.info(f"Success: {individual_id} -> {execution_time:.2f}s")
                else:
                    logger.warning(f"Failed to get valid response for {individual_id}")
                
            except Exception as e:
                logger.error(f"Unexpected error processing case {individual_id}: {e}")
                result = {
                    'id': individual_id,
                    'cause_of_death': 'PROCESSING_ERROR',
                    'code': 'ERROR',
                    'confidence': None,
                    'time_taken_seconds': 0,
                    'processed_at': datetime.now().isoformat()
                }
                results.append(result)
        
        # Calculate and log total processing time
        total_processing_time = time.time() - total_start_time
        avg_time = sum([r['time_taken_seconds'] for r in results]) / len(results) if results else 0
        
        logger.info(f"Total processing time: {total_processing_time:.2f} seconds")
        logger.info(f"Average time per case: {avg_time:.2f} seconds")
        
        return results
    

    
    def save_final_results(self, results: List[Dict], total_processing_time: float):
        """Save final results to CSV"""
        try:
            df_results = pd.DataFrame(results)
            
            # Calculate summary statistics
            successful_cases = len([r for r in results if r['cause_of_death'] not in ['ERROR', 'PROCESSING_ERROR']])
            avg_time = sum([r['time_taken_seconds'] for r in results]) / len(results) if results else 0
            
            
            summary_row = {
                'id': 'SUMMARY',
                'cause_of_death': f'Success Rate: {successful_cases}/{len(results)} ({successful_cases/len(results)*100:.1f}%)',
                'code': 'SUMMARY',
                'confidence': None,
                'time_taken_seconds': avg_time,
                'processed_at': f'Total Processing Time: {total_processing_time:.2f}s'
            }
            
            
            df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)
            
            # Ensure the output directory exists
            output_dir = Path(OUTPUT_CSV_PATH).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df_results.to_csv(OUTPUT_CSV_PATH, index=False)
            
            logger.info(f"Final results saved to {OUTPUT_CSV_PATH}")
            logger.info(f"Processed {len(results)} cases total")
            logger.info(f"Success rate: {successful_cases}/{len(results)} ({successful_cases/len(results)*100:.1f}%)")
            logger.info(f"Average processing time: {avg_time:.2f} seconds per case")
            logger.info(f"Total processing time: {total_processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")
            raise

    def load_valid_ids(self, clinician_cod_path: str) -> set:
        df_cod = pd.read_csv(clinician_cod_path)
        return set(df_cod['individual_id'].dropna().astype(str))

def main():
    """Main execution function"""
    logger.info("Starting Llama3:70b Few-Shot VA Analysis")
    logger.info(f"Input: {INPUT_CSV_PATH}")
    logger.info(f"Output: {OUTPUT_CSV_PATH}")
    logger.info(f"Model: {MODEL_NAME}")
    
    processor = LlamaVAProcessor()
    
    try:
        valid_ids = processor.load_valid_ids("/dataA/madiva/va/student/madiva_va_clinician_COD_20250926.csv")
        df = processor.load_dataset()
        df_filtered = df[df['individual_id'].astype(str).isin(valid_ids)]
        
       
        max_cases = None  # Process all cases
        
        
        logger.info("Starting LLM processing...")
        start_time = time.time()
        results = processor.process_cases(df_filtered, max_cases=max_cases)
        total_processing_time = time.time() - start_time

        processor.save_final_results(results, total_processing_time)
        
        logger.info("Processing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    
if __name__ == "__main__":
    main()