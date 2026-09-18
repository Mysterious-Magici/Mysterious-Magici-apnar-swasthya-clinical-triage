# Apnar Swasthya — Setup Guide (Beginner Friendly)

You now have a clinical triage app with **our own chatbot**. No Gemini, no OpenAI,
no API key — anyone who opens the link can chat with the bot immediately.

---

## Step 0 — What changed and why

| Before | Now |
|---|---|
| Chat needed a Gemini API key | Our own trained bot (`chatbot_model.pkl`) |
| Chat broke without internet | Works fully offline |
| No reset buttons | 3 reset buttons (parameters / chat / everything) |
| No export | PDF download + print button |
| Tab icon was 🩺 | Tab icon is the green Apnar Swasthya logo |

**Important honesty note:** the bot is *not* a language model like Gemini. Free
hosting (Streamlit Community Cloud) gives you about 1 GB of RAM, which cannot run
a real LLM. Instead your bot is a **trained intent classifier + clinical knowledge
engine**: it understands what the user is asking, then builds an answer from the
patient's actual numbers and a medical knowledge base you own. For a triage
assistant this is arguably *better* than an LLM — it can never hallucinate a drug
dose, and every answer is traceable to a file you control.

---

## Step 1 — Make your project folder

Put **all** of these in ONE folder (for example `apnar-swasthya`):

```
apnar-swasthya/
├── app.py                    ← NEW (replaces your old app.py)
├── chatbot_engine.py         ← NEW  (the bot's reply generator)
├── knowledge_base.py         ← NEW  (the medical content — edit this freely)
├── train_chatbot.py          ← NEW  (trains the bot)
├── pdf_report.py             ← NEW  (makes the PDF)
├── requirements.txt          ← UPDATED
├── train_model.py            ← your existing XGBoost trainer
├── xgboost_triage.pkl        ← your existing model
├── feature_encoders.pkl      ← your existing encoders
├── target_encoder.pkl        ← your existing encoder
├── logo.png
└── logo.jpeg
```

Delete the old `app.py` and paste the new one in its place.

---

## Step 2 — Install the libraries

Open a terminal **inside that folder** and run:

```bash
pip install -r requirements.txt
```

If `pip` is not recognised on Windows, use:

```bash
python -m pip install -r requirements.txt
```

---

## Step 3 — Train YOUR chatbot (one command, ~2 seconds)

```bash
python train_chatbot.py
```

You should see:

```
Training on 163 example phrases across 19 intents...
Training accuracy: 97.55%
Saved -> chatbot_model.pkl
```

A new file `chatbot_model.pkl` appears in the folder. **That file is your chatbot.**

> Re-run this command every time you add new questions to `train_chatbot.py`.

---

## Step 4 — (Optional) Re-train the triage model

Only needed if you get a scikit-learn version warning later:

```bash
python train_model.py
```

This regenerates the three `.pkl` files with your installed library versions.
It needs `INDIA_HEALTHCARE_DATA_2026.csv` in the same folder.

---

## Step 5 — Run the app

```bash
streamlit run app.py
```

Your browser opens `http://localhost:8501`. Check the sidebar:

- ✅ **Triage engine: XGBoost active**
- ✅ **Copilot: Apnar Swasthya offline bot**

If either shows a yellow warning, the matching `.pkl` file is missing from the folder.

---

## Step 6 — Try the three new features

**a) Reset buttons** — sidebar has three:
- *Clear Patient Parameters* → all sliders/boxes back to defaults
- *Clear Chat Conversation* → empties the chat (also available under the PDF button)
- *Clear Everything (New Patient)* → both at once

**b) PDF download** — scroll to **📄 Export Report** at the bottom right and click
*Download / Print Report (PDF)*. The PDF contains the parameters, the full
probability table, the whole conversation and the disclaimer. Open it and press
**Ctrl + P** to print.

**c) Logo as tab icon** — look at the browser tab; the green logo replaces 🩺.
This is done by `PAGE_ICON = Image.open("logo.png")` before `st.set_page_config`.
Streamlit needs the actual image object, which is why the code loads it first.

---

## Step 7 — Deploy free on Streamlit Community Cloud

1. Create a GitHub account and a **new public repository**, e.g. `apnar-swasthya`.
2. Upload every file from Step 1 — **including `chatbot_model.pkl` and the three
   triage `.pkl` files.** (Train locally, commit the result. The cloud does not run
   `train_chatbot.py` for you.)
3. Go to <https://share.streamlit.io> → **New app** → pick your repo → main file
   `app.py` → **Deploy**.
4. Wait 2–3 minutes. You get a public link like
   `https://apnar-swasthya.streamlit.app`.
5. Anyone who opens it can use the chatbot straight away — **no API key screen at all.**

> GitHub rejects files over 100 MB. Your `.pkl` files are ~1 MB, so you are fine.
> Do **not** upload the 25 MB CSV unless you want it public — the app does not need it.

---

## Step 8 — Make the bot smarter (the fun part)

**To add new medical content**, open `knowledge_base.py` and edit
`DISEASE_KB`. Each disease has: `overview`, `drivers`, `red_flags`, `tests`,
`management`, `lifestyle`, `prevention`, `recovery`, `referral`. Add a line, save,
refresh the browser. No re-training needed.

**To teach it new question phrasings**, open `train_chatbot.py`, add sentences to
the relevant intent list, for example:

```python
"tests": [
    "what tests should i order",
    "kaun sa test karna hai",      # add anything you like
],
```

Then run `python train_chatbot.py` again and restart the app.

**To add a brand-new kind of question**, add a new intent in `train_chatbot.py`,
then handle it in `chatbot_engine.py` inside `generate_reply()`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'fpdf'` | `pip install fpdf2` (note the **2**) |
| Sidebar says *Copilot: knowledge base only* | You skipped Step 3 — run `python train_chatbot.py` |
| Sidebar says *simulation mode* | The three triage `.pkl` files are not in the same folder as `app.py` |
| `InconsistentVersionWarning` from scikit-learn | Run `python train_model.py` to rebuild the `.pkl` files |
| Tab icon still shows 🩺 | `logo.png` is missing, or `pillow` is not installed |
| PDF shows odd characters | Expected — the PDF fonts are Latin-1; symbols like ×10⁹ are converted automatically |
| Bot answers "I could not match that" | Your phrasing is not in the training data yet — add it (Step 8) |

---

## Safety reminder for your demo/viva

Say this out loud in the presentation, and keep it visible in the app:

> Apnar Swasthya is a **decision-support prototype**. It ranks probabilities to help
> prioritise patients; it does not diagnose, does not prescribe, and does not replace
> a registered medical practitioner. The 97.7% accuracy figure describes held-out test
> data from one dataset, not real-world clinical performance.

Examiners always ask about this. Having the answer ready — plus the disclaimer printed
on every page of the PDF — usually scores better than the accuracy number itself.
