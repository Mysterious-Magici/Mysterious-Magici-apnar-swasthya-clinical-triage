"""
chatbot_engine.py
------------------------------------------------------------------
The reply generator for the Apnar Swasthya Clinical Copilot.

It takes:  the doctor's question + the current patient values + the
           triage probabilities
and returns a markdown answer.

No internet. No API key. Everything is computed locally from
chatbot_model.pkl (intent classifier) and knowledge_base.py.
------------------------------------------------------------------
"""

import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_base import (
    REFERENCE_RANGES,
    CRITICAL_RULES,
    DISEASE_KB,
    GENERAL_KB,
)

MODEL_FILE = "chatbot_model.pkl"

# Intents that need a disease to answer
DISEASE_INTENTS = {
    "why_diagnosis", "red_flags", "tests", "management",
    "lifestyle", "prevention", "recovery", "referral", "overview",
}

# Extra keyword rules. If any of these words appear, that intent wins
# outright - this makes short questions much more reliable.
KEYWORD_OVERRIDES = {
    "emergency": ["unconscious", "collapse", "not breathing", "cardiac arrest",
                  "seizure now", "cpr", "gasping", "unresponsive"],
    "why_diagnosis": ["why", "reason", "explain the prediction", "justify"],
    "red_flags": ["red flag", "danger", "warning sign", "alarm"],
    "tests": ["test", "investigation", "lab order", "imaging", "x-ray",
              "scan", "workup", "panel"],
    "management": ["treat", "treatment", "manage", "medicine", "medication",
                   "drug", "therapy", "dose"],
    "vitals": ["vital", "abnormal", "lab value", "interpret", "parameter",
               "reading", "numbers"],
    "lifestyle": ["diet", "food", "eat", "lifestyle", "exercise", "home care",
                  "nutrition"],
    "prevention": ["prevent", "avoid getting", "precaution", "vaccin",
                   "prophylaxis"],
    "recovery": ["recover", "prognosis", "how long", "outlook", "cure time"],
    "referral": ["refer", "admit", "admission", "hospitalis", "hospitaliz",
                 "escalate", "specialist"],
    "accuracy": ["accuracy", "accurate", "reliable", "trust", "dataset"],
    "privacy": ["privacy", "private", "store my data", "data safe"],
    "capabilities": ["what can you do", "help me", "your features"],
}


# ==================================================================
# LOADING
# ==================================================================
def load_chatbot(path: str = MODEL_FILE):
    """Load chatbot_model.pkl. Returns None if it has not been trained yet."""
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


# ==================================================================
# PATIENT ANALYSIS HELPERS
# ==================================================================
def analyse_patient(profile: dict):
    """Return (abnormal_findings, critical_findings) for the entered values."""
    abnormal, critical = [], []

    for field, (low, high, unit) in REFERENCE_RANGES.items():
        if field not in profile:
            continue
        value = float(profile[field])
        if value < low:
            abnormal.append(f"**{field}: {value} {unit}** - below the normal range ({low}-{high})")
        elif value > high:
            abnormal.append(f"**{field}: {value} {unit}** - above the normal range ({low}-{high})")

    for field, direction, limit, note in CRITICAL_RULES:
        if field not in profile:
            continue
        value = float(profile[field])
        if (direction == "below" and value < limit) or (direction == "above" and value > limit):
            critical.append(f"**{field} = {value}** -> {note}")

    return abnormal, critical


def detect_disease(text: str, fallback: str) -> str:
    """If the doctor names a disease in the question, answer about that one."""
    lowered = text.lower()
    aliases = {
        "Dengue": ["dengue", "ns1"],
        "Malaria": ["malaria", "plasmodium"],
        "Tuberculosis": ["tuberculosis", "tb", "koch"],
        "Common Cold": ["common cold", "cold", "flu", "viral fever"],
        "Hypertension": ["hypertension", "high bp", "blood pressure"],
        "Type-2 Diabetes": ["diabetes", "diabetic", "sugar", "hba1c"],
        "Cardiovascular Disease": ["cardiovascular", "heart attack", "cardiac",
                                   "coronary", "heart disease"],
        "Chronic Kidney Disease": ["kidney", "renal", "ckd", "nephro"],
        "Asthma/COPD": ["asthma", "copd", "wheez", "airway"],
        "Thyroid Disorder": ["thyroid", "tsh", "hypothyroid", "hyperthyroid"],
    }
    for disease, words in aliases.items():
        for word in words:
            if word in lowered:
                return disease
    return fallback


# ==================================================================
# INTENT DETECTION
# ==================================================================
def detect_intent(question: str, bundle) -> str:
    lowered = question.lower().strip()

    # 1. keyword shortcuts (fast and very reliable)
    for intent, words in KEYWORD_OVERRIDES.items():
        if any(word in lowered for word in words):
            # "why" alone should not beat a clear "what tests" question
            if intent == "why_diagnosis" and any(
                w in lowered for w in ["test", "treat", "diet", "refer"]
            ):
                continue
            return intent

    if bundle is None:
        return "fallback"

    # 2. trained classifier
    vec = bundle["vectorizer"].transform([lowered])
    probs = bundle["classifier"].predict_proba(vec)[0]
    best_idx = probs.argmax()
    intent = bundle["classifier"].classes_[best_idx]
    confidence = probs[best_idx]

    # 3. similarity safety check against the training phrases
    sims = cosine_similarity(vec, bundle["phrase_matrix"])[0]
    best_sim = float(sims.max())

    if confidence < 0.30 and best_sim < 0.25:
        return "fallback"
    return intent


# ==================================================================
# ANSWER BUILDERS
# ==================================================================
def _bullets(items):
    return "\n".join(f"- {item}" for item in items)


def _why(disease, profile, triage):
    kb = DISEASE_KB.get(disease)
    abnormal, critical = analyse_patient(profile)
    confidence = triage.get(disease, 0) * 100

    parts = [f"### Why the model points to {disease} ({confidence:.1f}% confidence)"]

    if abnormal:
        parts.append("**Values outside the normal range in this patient:**\n" + _bullets(abnormal[:10]))
    else:
        parts.append("Every entered vital and laboratory value is within the normal reference range, "
                     "so the model is working mainly from the demographic and borderline inputs. "
                     "Treat a prediction made on normal values with extra caution.")

    if kb:
        parts.append("**Patterns the model associates with " + disease + ":**\n" + _bullets(kb["drivers"]))

    ranked = sorted(triage.items(), key=lambda kv: kv[1], reverse=True)[:3]
    parts.append("**Ranked differentials:**\n" + _bullets(
        [f"{name} - {prob * 100:.1f}%" for name, prob in ranked]))

    if critical:
        parts.append("> **Critical values detected:**\n> " + "\n> ".join(critical))

    parts.append("_XGBoost reports a statistical association, not a causal explanation. "
                 "Confirm with history, examination and the tests listed below._")
    return "\n\n".join(parts)


def _red_flags(disease, profile, triage):
    kb = DISEASE_KB.get(disease)
    _, critical = analyse_patient(profile)
    parts = [f"### Red flags - {disease}"]

    if critical:
        parts.append("**Present in this patient right now:**\n" + _bullets(critical))
        parts.append("These need review before anything else on this page.")
    else:
        parts.append("No value entered has crossed a critical threshold, but keep monitoring.")

    if kb:
        parts.append("**Condition-specific warning signs to monitor:**\n" + _bullets(kb["red_flags"]))
        parts.append("**Escalation:** " + kb["referral"])
    return "\n\n".join(parts)


def _simple(disease, key, heading, closing=None):
    kb = DISEASE_KB.get(disease)
    if not kb:
        return GENERAL_KB["fallback"]
    body = kb[key]
    text = f"### {heading} - {disease}\n\n"
    text += _bullets(body) if isinstance(body, list) else body
    if closing:
        text += "\n\n" + closing
    return text


def _vitals(profile):
    abnormal, critical = analyse_patient(profile)
    parts = ["### Interpretation of the entered values"]
    if critical:
        parts.append("**Critical:**\n" + _bullets(critical))
    if abnormal:
        parts.append("**Outside the reference range:**\n" + _bullets(abnormal))
    else:
        parts.append("All entered vitals and laboratory values are within their normal reference ranges.")
    normal_count = len(REFERENCE_RANGES) - len(abnormal)
    parts.append(f"_{normal_count} of {len(REFERENCE_RANGES)} measured parameters are normal._")
    return "\n\n".join(parts)


# ==================================================================
# MAIN ENTRY POINT
# ==================================================================
def generate_reply(question: str, profile: dict, triage: dict, bundle=None) -> str:
    """
    question : what the user typed
    profile  : the patient dictionary from the form
    triage   : {disease: probability} from the XGBoost model
    bundle   : loaded chatbot_model.pkl (or None)
    """
    if not question or not question.strip():
        return GENERAL_KB["fallback"]

    top_disease = max(triage, key=triage.get) if triage else "Common Cold"
    disease = detect_disease(question, top_disease)
    intent = detect_intent(question, bundle)

    if intent == "why_diagnosis":
        return _why(disease, profile, triage)
    if intent == "red_flags":
        return _red_flags(disease, profile, triage)
    if intent == "tests":
        return _simple(disease, "tests", "Recommended investigations",
                       "_Order according to your institution's protocol and the patient's stability._")
    if intent == "management":
        return _simple(disease, "management", "Management principles",
                       "_These are general principles. Drug choice and dosing must be decided by "
                       "the treating physician._")
    if intent == "lifestyle":
        return _simple(disease, "lifestyle", "Diet and lifestyle counselling")
    if intent == "prevention":
        return _simple(disease, "prevention", "Prevention")
    if intent == "recovery":
        return _simple(disease, "recovery", "Expected course")
    if intent == "referral":
        return _simple(disease, "referral", "Referral and admission")
    if intent == "overview":
        return _simple(disease, "overview", "About")
    if intent == "vitals":
        return _vitals(profile)

    if intent in GENERAL_KB:
        return GENERAL_KB[intent]

    return GENERAL_KB["fallback"]
