import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Spot the Scam", layout="wide")

# Load saved model and preprocessor
model = joblib.load('models/logistic_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

st.title("🔍 Spot the Scam – Job Posting Fraud Detector")

# File upload
uploaded_file = st.file_uploader("Upload a job postings CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    raw_df = df.copy()

    # Handle text and categorical columns
    text_columns = ['company_profile', 'description', 'requirements', 'benefits']
    for col in text_columns:
        df[col] = df[col].fillna("")

    df['full_text'] = df['title'].fillna('') + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits']

    # Fill other categoricals
    cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    # Drop unused columns
    drop_cols = ['job_id', 'title', 'company_profile', 'description', 'requirements', 'benefits', 'salary_range', 'department']
    df = df.drop(columns=drop_cols, errors='ignore')

    # Predict
    X = preprocessor.transform(df)
    pred_probs = model.predict_proba(X)[:, 1]
    predictions = (pred_probs >= 0.5).astype(int)

    raw_df['Fraud Probability'] = pred_probs
    raw_df['Prediction'] = np.where(predictions == 1, "Fraud", "Genuine")

    # Show table
    st.subheader("📋 Prediction Results")
    st.dataframe(raw_df[['title', 'location', 'Fraud Probability', 'Prediction']].sort_values(by='Fraud Probability', ascending=False))

    # Pie Chart
    st.subheader("🧩 Fraud vs Genuine Jobs")
    pie_data = raw_df['Prediction'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=["red", "green"])
    ax1.axis('equal')
    st.pyplot(fig1)

    # Histogram
    st.subheader("📊 Fraud Probability Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(raw_df['Fraud Probability'], bins=20, kde=True, ax=ax2, color="orange")
    st.pyplot(fig2)

    # Top 10 suspicious
    st.subheader("🚩 Top 10 Most Suspicious Jobs")
    top10 = raw_df.sort_values(by='Fraud Probability', ascending=False).head(10)
    st.dataframe(top10[['title', 'location', 'Fraud Probability', 'Prediction']])
