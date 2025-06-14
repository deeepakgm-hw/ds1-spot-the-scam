import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import os

# --- Step 1: Load Preprocessed Data ---
X_train, y_train = joblib.load('models/train_data.pkl')
X_val, y_val = joblib.load('models/val_data.pkl')

# --- Step 2: Train Model ---
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# --- Step 3: Evaluate ---
y_pred = model.predict(X_val)
f1 = f1_score(y_val, y_pred)
print(f"✅ F1-score on validation set: {f1:.4f}")
print(classification_report(y_val, y_pred))

# --- Step 4: Save Model ---
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/logistic_model.pkl')
print("✅ Model saved to 'models/logistic_model.pkl'")
