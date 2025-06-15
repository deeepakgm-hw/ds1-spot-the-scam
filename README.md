# 🕵️‍♂️ Spot the Scam – Job Posting Fraud Detector

## 🚨 Why This Matters

Online job platforms are being targeted by scammers. These fake job listings waste time and risk exposing personal info. Our ML-based fraud detector helps job-seekers stay safe — **before they apply**.

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Streamlit
- Pandas, NumPy, Seaborn, Matplotlib

---

## ⚙️ Features

| Feature                       | Status |
|------------------------------|--------|
| Upload job posting CSV       | ✅ Done |
| Predict “Fraud” vs “Genuine” | ✅ Done |
| Show fraud probability       | ✅ Done |
| Interactive visualizations   | ✅ Done |
| Top-10 suspicious jobs       | ✅ Done |

---

## Create and activate a virtual environment (optional but recommended):
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Mac/Linux

## Install dependencies
pip install -r requirements.txt

## Run the Streamlit dashboard:
streamlit run src/dashboard.py

## 📊 Sample Dashboard
![image](https://github.com/user-attachments/assets/21da130d-9a8e-49ae-bde1-bc3402211b30)

The dashboard allows you to upload a CSV file of job postings and displays:

Predictions (Real/Fake)

Visualizations of fraud patterns

Fraud probability analysis

## 👨‍💻 Author
Deepak GM
Tharun Kumar SV




