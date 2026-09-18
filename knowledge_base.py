"""
knowledge_base.py
------------------------------------------------------------------
This file is the "brain content" of the Apnar Swasthya chatbot.

Nothing here calls the internet. It is plain Python data that the
chatbot reads to build its answers, so the app works offline and
needs NO API key of any kind.

Three things live here:
  1. REFERENCE_RANGES  -> normal ranges for every vital / lab field
  2. DISEASE_KB        -> clinical notes for each of the 10 diseases
                          that the XGBoost model can predict
  3. GENERAL_KB        -> answers that are not disease specific
------------------------------------------------------------------
"""

# ==================================================================
# 1. REFERENCE RANGES
#    (low, high, unit).  Used to flag which patient values are
#    abnormal, which is how the bot explains "why this diagnosis".
# ==================================================================
REFERENCE_RANGES = {
    "Body Temperature": (97.8, 99.1, "F"),
    "SpO2": (95.0, 100.0, "%"),
    "Respiratory Rate": (12.0, 20.0, "breaths/min"),
    "Heart Rate": (60.0, 100.0, "bpm"),
    "BMI": (18.5, 24.9, "kg/m2"),
    "Haemoglobin": (12.0, 17.5, "g/dL"),
    "Calcium": (8.5, 10.2, "mg/dL"),
    "Phosphorus": (2.5, 4.5, "mg/dL"),
    "Platelet Count": (150.0, 450.0, "x10^9/L"),
    "Cholesterol": (0.0, 200.0, "mg/dL"),
    "Sodium": (135.0, 145.0, "mEq/L"),
    "Potassium": (3.5, 5.0, "mIU/L"),
    "Neutrophils": (40.0, 70.0, "%"),
    "Lymphocytes": (20.0, 40.0, "%"),
    "CRP": (0.0, 1.0, "mg/dL"),
    "WBC Count": (4.5, 11.0, "x10^9/L"),
}

# Values outside these harder limits are treated as "critical"
CRITICAL_RULES = [
    ("SpO2", "below", 92.0, "Oxygen saturation below 92% - possible respiratory failure."),
    ("Platelet Count", "below", 50.0, "Severe thrombocytopenia - bleeding risk."),
    ("Body Temperature", "above", 104.0, "Hyperpyrexia - risk of seizure and organ stress."),
    ("Heart Rate", "above", 130.0, "Marked tachycardia - possible shock or sepsis."),
    ("Respiratory Rate", "above", 30.0, "Severe tachypnoea - respiratory distress."),
    ("Potassium", "above", 6.0, "Hyperkalaemia - cardiac arrhythmia risk."),
    ("Potassium", "below", 2.5, "Severe hypokalaemia - cardiac arrhythmia risk."),
    ("Sodium", "below", 125.0, "Severe hyponatraemia - risk of seizure."),
    ("Haemoglobin", "below", 7.0, "Severe anaemia - transfusion may be needed."),
    ("WBC Count", "below", 2.0, "Severe leucopenia - infection risk."),
]

# ==================================================================
# 2. DISEASE KNOWLEDGE BASE
#    Keys MUST match target_encoder.classes_ exactly.
# ==================================================================
DISEASE_KB = {
    "Dengue": {
        "overview": (
            "Dengue is a mosquito-borne viral infection spread by the Aedes aegypti mosquito. "
            "It is common across India during and after the monsoon. Classic features are high fever, "
            "severe headache, pain behind the eyes, joint and muscle pain, rash and bleeding tendency."
        ),
        "drivers": [
            "Falling platelet count (the single strongest signal in this dataset)",
            "High body temperature with a relatively low or normal white cell count",
            "Raised haematocrit / haemoglobin due to plasma leakage",
            "Raised CRP with neutrophil-lymphocyte shift",
        ],
        "red_flags": [
            "Platelets falling below 50 x10^9/L, or any active bleeding (gums, nose, stool, vomit)",
            "Severe abdominal pain or persistent vomiting",
            "Sudden drop in temperature with cold clammy skin - may signal the critical leakage phase",
            "Restlessness, drowsiness, reduced urine output, narrow pulse pressure (shock)",
        ],
        "tests": [
            "NS1 antigen (days 1-5 of fever), Dengue IgM/IgG (after day 5)",
            "Complete blood count with daily platelet and haematocrit monitoring",
            "Liver function tests (AST/ALT are often raised)",
            "Ultrasound abdomen / chest X-ray if leakage or effusion is suspected",
        ],
        "management": [
            "Supportive care - there is no specific antiviral treatment",
            "Aggressive oral fluids; IV crystalloids if the patient cannot drink or shows warning signs",
            "Paracetamol for fever; AVOID aspirin, ibuprofen and other NSAIDs (bleeding risk)",
            "Daily platelet and haematocrit trend is more important than a single value",
        ],
        "lifestyle": [
            "Rest and high fluid intake (ORS, coconut water, soups)",
            "Mosquito net or repellent even while ill, to prevent onward transmission",
            "Remove stagnant water around the home (coolers, pots, tyres)",
        ],
        "prevention": "Eliminate stagnant water, use repellents and full-sleeve clothing during daytime, and screen windows.",
        "recovery": "Fever usually settles in 5-7 days. The 24-48 hours after the fever breaks is the highest-risk window, so monitoring must continue.",
        "referral": "Admit immediately if there are warning signs, platelets under 50, bleeding, or any sign of shock.",
    },

    "Malaria": {
        "overview": (
            "Malaria is a parasitic infection transmitted by the Anopheles mosquito. "
            "It presents with cyclical fever with chills and rigors, sweating, headache, "
            "body aches, nausea and sometimes anaemia or an enlarged spleen."
        ),
        "drivers": [
            "High fever with chills, often in a periodic pattern",
            "Low platelet count together with anaemia (low haemoglobin)",
            "Raised CRP and altered neutrophil-lymphocyte ratio",
            "Normal or low white cell count despite high fever",
        ],
        "red_flags": [
            "Altered consciousness, confusion or seizures (cerebral malaria)",
            "Jaundice, very dark urine, or falling urine output",
            "Breathlessness or oxygen saturation below 94%",
            "Severe anaemia (haemoglobin under 7 g/dL) or spontaneous bleeding",
        ],
        "tests": [
            "Peripheral blood smear (thick and thin film) - the reference standard",
            "Rapid diagnostic test (RDT) for P. falciparum / P. vivax antigen",
            "Complete blood count, liver and renal function, blood glucose",
            "Parasite density count to grade severity",
        ],
        "management": [
            "Species-specific antimalarial therapy per the current National Vector Borne Disease Control Programme guidance",
            "Artemisinin-based combination therapy for falciparum; chloroquine plus primaquine for vivax where sensitive",
            "Antipyretics, fluids and correction of hypoglycaemia",
            "Severe malaria needs parenteral artesunate and inpatient care",
        ],
        "lifestyle": [
            "Complete the full course of medication even after fever settles",
            "Sleep under an insecticide-treated bed net",
            "Return immediately if fever recurs after treatment (relapse is possible in vivax)",
        ],
        "prevention": "Insecticide-treated nets, indoor residual spraying, and removing mosquito breeding sites near the home.",
        "recovery": "Fever typically clears in 48-72 hours of correct therapy. Fatigue and anaemia may take a few weeks to resolve.",
        "referral": "Any sign of severe or cerebral malaria requires immediate hospital admission.",
    },

    "Tuberculosis": {
        "overview": (
            "Tuberculosis is a chronic bacterial infection caused by Mycobacterium tuberculosis, "
            "most often affecting the lungs. India carries the largest TB burden globally. "
            "Typical features are a cough lasting more than two weeks, evening fever, night sweats and weight loss."
        ),
        "drivers": [
            "Prolonged low-grade fever with chronic cough",
            "Lymphocyte predominance on the differential count",
            "Raised CRP with a long symptom duration",
            "Low BMI and anaemia, reflecting chronic disease",
        ],
        "red_flags": [
            "Coughing up blood (haemoptysis)",
            "Breathlessness at rest or oxygen saturation below 92%",
            "Rapid unintentional weight loss",
            "Neck stiffness, headache or confusion (possible TB meningitis)",
        ],
        "tests": [
            "Sputum for CBNAAT / GeneXpert MTB-RIF (also detects rifampicin resistance)",
            "Chest X-ray, and CT chest if the X-ray is inconclusive",
            "Sputum smear microscopy and culture with drug susceptibility testing",
            "HIV test and blood sugar screening, as both alter management",
        ],
        "management": [
            "Standard multi-drug therapy under the National TB Elimination Programme (free of cost in India)",
            "Intensive phase followed by continuation phase - typically 6 months for drug-sensitive TB",
            "Directly observed treatment support and contact tracing of household members",
            "Monthly weight and adherence monitoring; liver function if symptoms of drug toxicity appear",
        ],
        "lifestyle": [
            "Never stop treatment early - incomplete courses drive drug resistance",
            "High-protein, high-calorie diet to regain weight",
            "Cover the mouth while coughing and keep rooms well ventilated for the first weeks",
        ],
        "prevention": "BCG vaccination in infancy, good ventilation, early diagnosis of contacts, and preventive therapy for close contacts.",
        "recovery": "Most patients feel better within 2-4 weeks, but the full course must be completed to achieve cure.",
        "referral": "Refer to a designated microscopy centre or DOTS centre for free diagnosis and treatment.",
    },

    "Common Cold": {
        "overview": (
            "The common cold is a self-limiting viral upper respiratory infection. "
            "It causes a runny or blocked nose, sneezing, sore throat, mild cough, "
            "mild fever and general tiredness, usually settling within a week."
        ),
        "drivers": [
            "Mild temperature elevation rather than high fever",
            "Near-normal platelet and white cell counts",
            "Only mildly raised CRP",
            "Normal oxygen saturation and respiratory rate",
        ],
        "red_flags": [
            "Fever lasting more than 5 days or rising again after improving",
            "Breathlessness, chest pain or oxygen saturation below 94%",
            "Severe one-sided facial pain or a swollen, painful throat that limits swallowing",
            "Symptoms persisting beyond 10-14 days",
        ],
        "tests": [
            "Usually no investigation is needed for a straightforward cold",
            "Consider a complete blood count and CRP if fever is prolonged",
            "Rapid influenza or COVID-19 testing during outbreaks",
            "Chest X-ray only if lower respiratory infection is suspected",
        ],
        "management": [
            "Symptomatic care - rest, warm fluids, steam inhalation, saline nasal drops",
            "Paracetamol for fever or body ache",
            "Antibiotics are NOT indicated for a viral cold",
            "Review if symptoms worsen or do not improve in a week",
        ],
        "lifestyle": [
            "Adequate sleep and hydration",
            "Hand hygiene to avoid spreading the infection at home or work",
            "Avoid smoking and smoky environments while recovering",
        ],
        "prevention": "Frequent handwashing, avoiding close contact with symptomatic people, and annual flu vaccination for high-risk groups.",
        "recovery": "Typically 5-7 days; a dry cough can linger for two to three weeks.",
        "referral": "Routine outpatient care is enough unless red flags appear.",
    },

    "Hypertension": {
        "overview": (
            "Hypertension is persistently raised blood pressure. It is usually symptomless, "
            "which is why it is called a silent condition, but it is a leading driver of stroke, "
            "heart attack and kidney disease in India."
        ),
        "drivers": [
            "Raised heart rate together with an elevated BMI",
            "Raised cholesterol",
            "Age above 40 years",
            "Sodium and potassium values at the edge of the normal range",
        ],
        "red_flags": [
            "Blood pressure at or above 180/120 with symptoms - a hypertensive emergency",
            "Chest pain, severe breathlessness or sudden severe headache",
            "Sudden weakness, facial droop or slurred speech (stroke)",
            "Blurred vision or confusion",
        ],
        "tests": [
            "Repeat office BP readings plus home or 24-hour ambulatory monitoring",
            "ECG and echocardiogram to look for cardiac strain",
            "Renal function, electrolytes and urine protein",
            "Lipid profile, fasting glucose and HbA1c for overall cardiovascular risk",
        ],
        "management": [
            "Lifestyle modification is first-line and continues alongside any drug therapy",
            "Salt reduction to under 5 g per day and weight reduction",
            "Antihypertensive medication chosen by the treating physician based on age, comorbidity and BP level",
            "Regular home BP monitoring with a written log",
        ],
        "lifestyle": [
            "Reduce salt, pickles, papad and packaged snacks",
            "30 minutes of brisk walking most days of the week",
            "Stop tobacco and limit alcohol",
            "Manage stress and aim for 7-8 hours of sleep",
        ],
        "prevention": "Maintain a healthy weight, keep salt low, stay physically active and get BP checked at least yearly after age 30.",
        "recovery": "Hypertension is controlled rather than cured; treatment is usually lifelong and should not be stopped once BP normalises.",
        "referral": "Immediate emergency care for any hypertensive emergency; otherwise routine physician follow-up.",
    },

    "Type-2 Diabetes": {
        "overview": (
            "Type-2 diabetes is a chronic metabolic disorder in which blood glucose stays high "
            "because the body resists insulin. Common features are excessive thirst, frequent urination, "
            "tiredness, blurred vision, slow wound healing and unexplained weight change."
        ),
        "drivers": [
            "Elevated BMI",
            "Raised cholesterol",
            "Age and long-standing metabolic strain",
            "Mildly raised CRP indicating low-grade inflammation",
        ],
        "red_flags": [
            "Very high glucose with vomiting, rapid deep breathing or fruity breath (ketoacidosis)",
            "Confusion, extreme drowsiness or dehydration",
            "A foot ulcer, blackened toe or spreading foot infection",
            "Sudden vision loss",
        ],
        "tests": [
            "Fasting and post-prandial blood glucose",
            "HbA1c for three-month average control",
            "Lipid profile, renal function and urine albumin-creatinine ratio",
            "Annual dilated eye examination and foot examination",
        ],
        "management": [
            "Medical nutrition therapy and structured physical activity",
            "Oral glucose-lowering medication, and insulin where indicated, as prescribed by a physician",
            "Home glucose monitoring with a record book",
            "Screen and treat blood pressure and cholesterol alongside glucose",
        ],
        "lifestyle": [
            "Portion-controlled meals with whole grains, dal, vegetables; reduce white rice, sugar and sweets",
            "150 minutes of moderate exercise per week",
            "Daily foot inspection and well-fitting footwear",
            "Weight reduction of 5-10% produces a large improvement in control",
        ],
        "prevention": "Healthy weight, active lifestyle, reduced refined carbohydrate intake, and screening from age 30 if there is a family history.",
        "recovery": "Long-term control is the goal; early intensive lifestyle change can bring some patients into remission.",
        "referral": "Endocrinology or diabetology referral if control remains poor or complications appear.",
    },

    "Cardiovascular Disease": {
        "overview": (
            "Cardiovascular disease covers conditions affecting the heart and blood vessels, "
            "including coronary artery disease and heart failure. Warning features are chest pain "
            "or pressure, breathlessness, palpitations, cold sweat, leg swelling and pain radiating "
            "to the arm, jaw or back."
        ),
        "drivers": [
            "Abnormal heart rate with breathlessness",
            "Raised cholesterol and BMI",
            "Older age",
            "Raised CRP as a vascular inflammation marker",
        ],
        "red_flags": [
            "Chest pain or pressure lasting more than 10-15 minutes, or at rest",
            "Pain radiating to the left arm, jaw, neck or back with sweating and nausea",
            "Sudden severe breathlessness, especially lying flat or at night",
            "Fainting, very slow or very fast pulse, or oxygen saturation below 92%",
        ],
        "tests": [
            "12-lead ECG immediately - within 10 minutes for suspected acute coronary syndrome",
            "Troponin I or T, repeated as per protocol",
            "Echocardiogram and chest X-ray",
            "Lipid profile, blood glucose, renal function; stress test or angiography as advised",
        ],
        "management": [
            "Suspected heart attack is a time-critical emergency - arrange immediate transfer to a facility with cardiac care",
            "Oxygen if saturation is low, and monitoring of rhythm and blood pressure",
            "Definitive therapy (thrombolysis, angioplasty, medication) is decided by the cardiology team",
            "Long-term secondary prevention with medication, diet and supervised activity",
        ],
        "lifestyle": [
            "Complete tobacco cessation",
            "Low-salt, low-saturated-fat diet with more vegetables and fibre",
            "Graded physical activity as advised after the acute event",
            "Strict control of blood pressure, sugar and cholesterol",
        ],
        "prevention": "Control blood pressure, glucose and lipids; stay active; avoid tobacco; maintain a healthy weight.",
        "recovery": "Depends on the severity of the event; cardiac rehabilitation greatly improves outcomes.",
        "referral": "Do not delay - call emergency services for suspected acute coronary syndrome.",
    },

    "Chronic Kidney Disease": {
        "overview": (
            "Chronic kidney disease is a gradual, long-term loss of kidney function. "
            "It often follows diabetes or hypertension and may show swelling of the feet or face, "
            "reduced or frothy urine, tiredness, nausea and itching."
        ),
        "drivers": [
            "Abnormal sodium, potassium, calcium and phosphorus - the classic electrolyte pattern of CKD",
            "Low haemoglobin (anaemia of chronic kidney disease)",
            "Raised phosphorus with low calcium",
            "Associated hypertension and older age",
        ],
        "red_flags": [
            "Potassium above 6.0 mEq/L - risk of dangerous heart rhythms",
            "Very little or no urine output",
            "Severe breathlessness from fluid overload",
            "Confusion, persistent vomiting or seizures (uraemia)",
        ],
        "tests": [
            "Serum creatinine with estimated GFR to stage the disease",
            "Urine routine and albumin-creatinine ratio",
            "Electrolytes, calcium, phosphorus, parathyroid hormone",
            "Ultrasound of the kidneys and urinary tract; haemoglobin for anaemia",
        ],
        "management": [
            "Treat the underlying cause - tight control of diabetes and blood pressure",
            "Dietary sodium, potassium, phosphorus and protein adjustment guided by a renal dietitian",
            "Avoid nephrotoxic drugs, especially NSAID painkillers and unsupervised contrast studies",
            "Manage anaemia and bone-mineral disturbance; plan dialysis or transplant in advanced stages",
        ],
        "lifestyle": [
            "Strictly limit salt; avoid packaged and preserved food",
            "Follow the prescribed fluid allowance",
            "Never take over-the-counter painkillers without checking with the treating doctor",
        ],
        "prevention": "Control diabetes and blood pressure, stay hydrated, avoid unsupervised painkillers, and screen annually if at risk.",
        "recovery": "CKD is usually progressive; the goal is to slow progression and protect remaining function.",
        "referral": "Nephrology referral for stage 3 or worse, rapidly rising creatinine, or dangerous electrolyte values.",
    },

    "Asthma/COPD": {
        "overview": (
            "Asthma and COPD are chronic airway diseases causing breathlessness, wheeze, "
            "chest tightness and cough. Asthma is typically variable and often allergic; "
            "COPD is largely irreversible and strongly linked to smoking and biomass smoke exposure."
        ),
        "drivers": [
            "Low oxygen saturation",
            "Raised respiratory rate",
            "Raised heart rate from respiratory effort",
            "Eosinophil or lymphocyte shift on the differential count",
        ],
        "red_flags": [
            "Oxygen saturation below 92% on room air",
            "Unable to complete a full sentence in one breath",
            "A silent chest, drowsiness or confusion - impending respiratory failure",
            "No relief after repeated reliever inhaler doses",
        ],
        "tests": [
            "Spirometry with bronchodilator reversibility - the key diagnostic test",
            "Peak expiratory flow monitoring",
            "Chest X-ray to exclude infection or pneumothorax",
            "Arterial blood gas in severe exacerbation; blood eosinophil count",
        ],
        "management": [
            "Controlled oxygen to reach the target saturation",
            "Inhaled bronchodilators via spacer or nebuliser as prescribed",
            "Inhaled corticosteroid maintenance therapy for asthma; systemic steroids for exacerbations as directed",
            "Correct inhaler technique is as important as the drug itself - demonstrate and re-check it",
        ],
        "lifestyle": [
            "Stop smoking and avoid chulha or biomass smoke exposure",
            "Identify and avoid triggers - dust, pollen, cold air, strong smells",
            "Annual influenza and pneumococcal vaccination",
            "Keep a written action plan and always carry the reliever inhaler",
        ],
        "prevention": "Avoid tobacco and indoor smoke, improve kitchen ventilation, and treat allergies early.",
        "recovery": "Exacerbations typically settle over days with treatment, but maintenance therapy must continue.",
        "referral": "Emergency care for a severe attack; pulmonology referral for frequent exacerbations.",
    },

    "Thyroid Disorder": {
        "overview": (
            "Thyroid disorders involve an underactive (hypothyroid) or overactive (hyperthyroid) gland. "
            "Hypothyroidism causes fatigue, weight gain, cold intolerance, dry skin, hair loss and "
            "menstrual changes; hyperthyroidism causes weight loss, palpitations, heat intolerance and tremor."
        ),
        "drivers": [
            "Heart rate outside the normal range in either direction",
            "BMI change with fatigue",
            "Cholesterol elevation, which is common in hypothyroidism",
            "Non-specific inflammatory markers",
        ],
        "red_flags": [
            "Very slow pulse with low temperature and drowsiness (myxoedema coma)",
            "Very fast irregular pulse with fever and agitation (thyroid storm)",
            "A rapidly enlarging neck swelling, hoarseness or difficulty swallowing",
            "Severe palpitation with chest pain",
        ],
        "tests": [
            "TSH as the first-line screening test",
            "Free T4 and Free T3",
            "Thyroid peroxidase antibodies where autoimmune disease is suspected",
            "Thyroid ultrasound for nodules or goitre",
        ],
        "management": [
            "Hypothyroidism: daily levothyroxine on an empty stomach, dose titrated to TSH",
            "Hyperthyroidism: antithyroid drugs, radioiodine or surgery as decided by the physician",
            "Repeat TSH 6-8 weeks after any dose change",
            "Review any medicines or supplements (iron, calcium) that interfere with absorption",
        ],
        "lifestyle": [
            "Take levothyroxine at the same time daily, 30-60 minutes before food",
            "Use iodised salt",
            "Keep at least a 4-hour gap from iron or calcium supplements",
        ],
        "prevention": "Use iodised salt and screen during pregnancy or where there is a family history.",
        "recovery": "Symptoms usually improve over 6-8 weeks of adequate treatment, which is often lifelong.",
        "referral": "Endocrinology referral for nodules, pregnancy, or unstable thyroid function.",
    },
}

# ==================================================================
# 3. GENERAL (NON-DISEASE) ANSWERS
# ==================================================================
GENERAL_KB = {
    "greeting": (
        "Hello. I am the Apnar Swasthya Clinical Copilot, running fully offline on this device. "
        "I can explain the triage result, list red flags, suggest confirmatory tests, outline "
        "management principles and interpret the entered vitals. What would you like to review?"
    ),
    "capabilities": (
        "I can help with:\n\n"
        "- **Why this result** - which vitals and labs pushed the model toward this condition\n"
        "- **Red flags** - what needs urgent attention right now\n"
        "- **Tests** - confirmatory laboratory and imaging orders\n"
        "- **Management** - standard treatment principles\n"
        "- **Vitals** - a plain reading of every abnormal value entered\n"
        "- **Diet and lifestyle**, **prevention**, **recovery time**, and **when to refer**\n\n"
        "Ask in plain English, or tap one of the quick buttons above the chat box."
    ),
    "thanks": "Happy to help. Remember to document the findings in the patient record before the shift change.",
    "goodbye": "Take care. Clear the session before entering the next patient's details.",
    "accuracy": (
        "The triage model is an XGBoost classifier trained on the INDIA_HEALTHCARE_DATA_2026 dataset "
        "of 100,000 records across 10 conditions, reaching roughly 97.7% accuracy on a held-out test set. "
        "That number describes performance on data resembling the training set, not a guarantee for any "
        "individual patient. Treat the output as a prioritisation aid only - the final diagnosis must come "
        "from a licensed clinician with history, examination and confirmatory tests."
    ),
    "how_it_works": (
        "There are two separate parts. First, an XGBoost classifier takes the 21 entered fields, "
        "encodes the categorical ones and returns a probability for each of the 10 conditions. "
        "Second, I compare every entered value against standard reference ranges and match your "
        "question against a locally stored clinical knowledge base. No data leaves this machine and "
        "no API key is used."
    ),
    "privacy": (
        "Nothing you type is sent anywhere. The model files and the knowledge base sit in the "
        "application folder, and all processing happens in the running Streamlit process. "
        "No patient identifiers are collected by the form by design."
    ),
    "emergency": (
        "If the patient is unstable - unresponsive, struggling to breathe, oxygen saturation below 92%, "
        "chest pain, active bleeding or signs of shock - stop using this tool and escalate to emergency "
        "care immediately. In India, dial 108 for an ambulance.\n\n"
        "This application cannot assess an emergency and must never delay direct clinical care."
    ),
    "disclaimer": (
        "Apnar Swasthya is a decision-support and educational prototype. It does not diagnose, does not "
        "prescribe, and does not replace examination by a registered medical practitioner. Every output "
        "must be confirmed clinically before any treatment decision."
    ),
    "fallback": (
        "I could not match that to anything in my knowledge base. I work best on questions about the "
        "current triage result - for example: *why this diagnosis*, *what are the red flags*, "
        "*which tests should I order*, *how is it managed*, *explain the vitals*, *what diet should the "
        "patient follow*, or *when should I refer*. You can also ask about a specific condition by name, "
        "such as \"tell me about tuberculosis\"."
    ),
}
