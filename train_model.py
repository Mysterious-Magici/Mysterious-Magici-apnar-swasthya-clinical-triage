import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading data...")
# 1. Define required features
features = [
    'Age', 'Gender', 'Socio-economic Status', 'Insurance', 'Hospital Type',
    'Body Temperature', 'SpO2', 'Respiratory Rate', 'Heart Rate', 'BMI',
    'Haemoglobin', 'Calcium', 'Phosphorus', 'Platelet Count', 'Cholesterol',
    'Sodium', 'Potassium', 'Neutrophils', 'Lymphocytes', 'CRP', 'WBC Count'
]

# 2. Load dataset and drop missing values
df = pd.read_csv("INDIA_HEALTHCARE_DATA_2026.csv").dropna(subset=features + ['Diseases'])
X = df[features].copy()
y = df['Diseases']

print("Encoding features...")
# 3. Encode Categorical Features safely
encoders = {}
for col in ['Gender', 'Socio-economic Status', 'Insurance', 'Hospital Type']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# 4. Encode Target Variable
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Training XGBoost model...")
# 6. Train the Model using optimized parameters
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(target_encoder.classes_),
    eval_metric='mlogloss',
    learning_rate=0.1,
    max_depth=4,
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# 7. Generate Fresh Binary Pickle Files
joblib.dump(model, 'xgboost_triage.pkl')
joblib.dump(encoders, 'feature_encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

print(" Success! Fresh .pkl files generated safely in your root directory.")