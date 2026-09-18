"""
train_chatbot.py
------------------------------------------------------------------
Trains the Apnar Swasthya chatbot's intent classifier and saves it
as  chatbot_model.pkl.

How it works (simple version):
  * Below there is a list of example sentences for every "intent"
    (an intent = a type of question a doctor might ask).
  * We convert those sentences into numbers using TF-IDF.
  * We train a Logistic Regression classifier on them.
  * At chat time the app converts the user's question the same way,
    and the classifier + a cosine-similarity check decide which
    intent it belongs to.

Run it once:      python train_chatbot.py
Re-run it any time you add more example sentences.
------------------------------------------------------------------
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==================================================================
# TRAINING DATA - add more phrases here any time to make the bot smarter
# ==================================================================
INTENT_TRAINING_DATA = {
    "why_diagnosis": [
        "why this diagnosis",
        "why did the model predict this",
        "what led to this prediction",
        "explain the prediction",
        "which vitals caused this result",
        "what lab anomalies support this condition",
        "justify the triage result",
        "reason for this disease prediction",
        "how did you reach this conclusion",
        "what features drove the model",
        "why is it showing this condition",
        "basis of this assessment",
    ],
    "red_flags": [
        "what are the red flags",
        "any danger signs",
        "critical risks to monitor",
        "what should i watch out for",
        "warning signs for this patient",
        "is this patient critical",
        "emergent risks right now",
        "what can go wrong",
        "complications to look for",
        "how serious is this case",
        "danger signals",
        "should i be worried about this patient",
    ],
    "tests": [
        "what tests should i order",
        "recommended investigations",
        "confirmatory laboratory tests",
        "which imaging is needed",
        "next diagnostic steps",
        "what blood work is required",
        "how do i confirm this diagnosis",
        "standard protocol investigations",
        "lab orders for this condition",
        "which panel should i send",
        "diagnostic workup",
    ],
    "management": [
        "how is it treated",
        "what is the treatment",
        "management plan",
        "which medicines are used",
        "therapy for this condition",
        "how do i manage this patient",
        "treatment protocol",
        "what drugs should be given",
        "clinical management steps",
        "what is the standard of care",
        "how to treat this disease",
    ],
    "vitals": [
        "explain the vitals",
        "interpret the lab values",
        "which values are abnormal",
        "what do these numbers mean",
        "summarise the patient parameters",
        "read the blood report",
        "are the labs normal",
        "analyse the entered values",
        "what is abnormal in this patient",
        "vital sign interpretation",
    ],
    "lifestyle": [
        "what diet should the patient follow",
        "lifestyle advice",
        "what food is recommended",
        "home care instructions",
        "what should the patient avoid",
        "daily routine advice",
        "nutrition guidance",
        "exercise recommendation",
        "self care at home",
        "patient counselling points",
    ],
    "prevention": [
        "how can this be prevented",
        "prevention measures",
        "how to avoid getting this",
        "prophylaxis advice",
        "what precautions should the family take",
        "how to stop it spreading",
        "preventive steps",
        "vaccination advice",
    ],
    "recovery": [
        "how long will recovery take",
        "prognosis of this condition",
        "when will the patient feel better",
        "expected recovery time",
        "will this patient recover",
        "how many days for cure",
        "what is the outlook",
        "duration of illness",
    ],
    "referral": [
        "should i refer this patient",
        "when to admit",
        "does the patient need hospitalisation",
        "which specialist should see this patient",
        "escalation criteria",
        "when should i send to higher centre",
        "is admission required",
        "referral advice",
    ],
    "overview": [
        "tell me about this disease",
        "what is this condition",
        "give me an overview",
        "explain this illness",
        "describe the disease",
        "background of this condition",
        "what are the usual symptoms",
        "typical presentation",
        "general information about this disease",
    ],
    "greeting": [
        "hello",
        "hi there",
        "good morning",
        "hey",
        "namaste",
        "good evening doctor",
        "hi bot",
        "are you there",
    ],
    "thanks": [
        "thank you",
        "thanks a lot",
        "appreciate it",
        "that was helpful",
        "great answer thanks",
        "thank you so much",
    ],
    "goodbye": [
        "bye",
        "goodbye",
        "see you later",
        "that is all",
        "ending session",
        "i am done",
    ],
    "capabilities": [
        "what can you do",
        "help",
        "how can you assist me",
        "what are your features",
        "list your abilities",
        "what questions can i ask",
        "how do i use this",
        "guide me",
    ],
    "accuracy": [
        "how accurate is the model",
        "can i trust this prediction",
        "what is the model accuracy",
        "how reliable is this",
        "is this prediction correct",
        "what dataset was used",
        "how was the model trained",
        "model performance",
    ],
    "how_it_works": [
        "how does this app work",
        "what algorithm do you use",
        "explain the system",
        "which model is behind this",
        "do you use xgboost",
        "how do you generate answers",
        "are you using chatgpt",
        "do you need an api key",
    ],
    "privacy": [
        "is patient data safe",
        "do you store my data",
        "is this private",
        "where does the data go",
        "data privacy",
        "do you send information online",
    ],
    "emergency": [
        "this is an emergency",
        "patient is collapsing",
        "patient is unconscious",
        "patient stopped breathing",
        "severe bleeding right now",
        "urgent help needed",
        "patient is in shock",
        "cardiac arrest",
    ],
    "disclaimer": [
        "is this medical advice",
        "are you a doctor",
        "legal disclaimer",
        "can i rely on this for treatment",
        "should i follow this blindly",
        "what are the limitations",
    ],
}


def build_and_save(path: str = "chatbot_model.pkl") -> None:
    texts, labels = [], []
    for intent, phrases in INTENT_TRAINING_DATA.items():
        for phrase in phrases:
            texts.append(phrase.lower())
            labels.append(intent)

    print(f"Training on {len(texts)} example phrases across "
          f"{len(INTENT_TRAINING_DATA)} intents...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        stop_words="english",   # ignore filler words like 'what', 'the', 'is'
    )
    X = vectorizer.fit_transform(texts)

    classifier = LogisticRegression(max_iter=1000, C=8.0)
    classifier.fit(X, labels)

    bundle = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "phrase_matrix": X,          # used for the similarity safety check
        "phrase_labels": labels,
    }
    joblib.dump(bundle, path)

    train_acc = classifier.score(X, labels)
    print(f"Training accuracy: {train_acc:.2%}")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    build_and_save()
