import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

# Load raw data
df = pd.read_csv(r'C:\Users\deepu\Downloads\ds1-spot-the-scamdatatrain.csv.csv')

# --- Step 1: Fill Nulls for Text Columns ---
text_columns = ['company_profile', 'description', 'requirements', 'benefits']
for col in text_columns:
    df[col] = df[col].fillna("")

# Combine into a single text field
df['full_text'] = df['title'].fillna('') + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits']

# --- Step 2: Drop Useless or Redundant Columns ---
drop_cols = ['job_id', 'title', 'company_profile', 'description', 'requirements', 'benefits', 'salary_range', 'department']
df = df.drop(columns=drop_cols, errors='ignore')

# --- Step 3: Handle Categorical Columns ---
cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

# --- Step 4: Split Features and Labels ---
X = df.drop(columns=['fraudulent'])
y = df['fraudulent']

# --- Step 5: Preprocessing Pipelines ---

# Text pipeline
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000))
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer
preprocessor = ColumnTransformer([
    ('text', text_pipeline, 'full_text'),
    ('cat', cat_pipeline, cat_cols)
])

# --- Step 6: Fit & Transform ---
X_processed = preprocessor.fit_transform(X)

# Save preprocessor for reuse
os.makedirs('models', exist_ok=True)
joblib.dump(preprocessor, 'models/preprocessor.pkl')

# --- Step 7: Train/Test Split ---
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=42)

# Save processed sets
joblib.dump((X_train, y_train), 'models/train_data.pkl')
joblib.dump((X_val, y_val), 'models/val_data.pkl')

print("✅ Preprocessing complete. Data saved to 'models/' folder.")

