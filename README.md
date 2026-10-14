#  Apnar Swasthya: Multi-Disease Predictive Diagnostic System

**Live Application:** 
Stremlit Cloud: [https://apnar-swasthya-app.streamlit.app/](https://apnar-swasthya-app.streamlit.app/), 
Render Cloud: [https://apnar-swasthya.onrender.com](https://apnar-swasthya.onrender.com)
![Preview](https://github.com/Mysterious-Magici/Mysterious-Magici-apnar-swasthya-clinical-triage/blob/main/Apnar%20Swathya.png)

##  Overview
Apnar Swasthya is an intelligent clinical decision support system designed to assist healthcare professionals in patient triage. By analyzing patient demographics, physiological vitals, and comprehensive laboratory markers, the application utilizes advanced machine learning classification to predict potential differential diagnoses and stratify clinical risk in real-time.

##  Key Features
*   **Comprehensive Clinical Intake:** Captures physiological vitals (SpO2, Heart Rate, etc.) and detailed laboratory markers (Platelets, CRP, Neutrophils, Lymphocytes).
*   **Predictive Diagnostic Engine:** Utilizes a trained XGBoost classifier to compute confidence probabilities for multiple conditions (e.g., Dengue, Type-2 Diabetes, Thyroid Disorders).
*   **Automated Risk Stratification:** Integrates algorithmic clinical guardrails to flag critical thresholds (e.g., severe thrombocytopenia) for immediate medical review.
*   **Professional PDF Reporting:** Automatically generates downloadable, formatted clinical triage assessment reports for physician review and record-keeping.
*   **Interactive Visualizations:** Dynamically visualizes model confidence probability scores using Altair.

##  Technology Stack
*   **Frontend & Web Framework:** Streamlit
*   **Machine Learning:** XGBoost, Scikit-Learn, Joblib
*   **Data Processing:** Python, Pandas, NumPy
*   **Visualization & Export:** Altair, FPDF
*   **Cloud Deployment:** Render

##  Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Mysterious-Magici/Mysterious-Magici-apnar-swasthya-clinical-triage.git](https://github.com/Mysterious-Magici/Mysterious-Magici-apnar-swasthya-clinical-triage.git)
   cd Mysterious-Magici-apnar-swasthya-clinical-triage

2. **Install the required dependencies:**
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt

```


3. **Launch the application:**
```bash
streamlit run app.py

```


*(Note: The system will be hosted locally at `http://localhost:8501`)*

##  Medical Disclaimer

**For Educational & Clinical Decision Support Purposes Only.**
This application is designed as an automated triage and risk stratification tool. It is *not* a substitute for professional medical advice, diagnosis, or treatment. All algorithmic outputs and findings must be rigorously evaluated and verified by a licensed healthcare professional prior to clinical disposition.

```

```
