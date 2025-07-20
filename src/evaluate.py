<<<<<<< HEAD
import sys
import json
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def evaluate_model(data_path, model_path, metric_path):
    df = pd.read_csv(data_path)
    X = df.drop("Loan_Status_Y", axis=1)
    y = df["Loan_Status_Y"]

    model = joblib.load(model_path)
    preds = model.predict(X)

    accuracy = accuracy_score(y, preds)

    metrics = {
        "accuracy": accuracy
    }

    with open(metric_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[✅] Evaluation complete. Accuracy: {accuracy:.4f}")
    print(f"[📄] Metrics written to {metric_path}")

if __name__ == "__main__":
    evaluate_model(sys.argv[1], sys.argv[2], sys.argv[3])
=======
# src/evaluate.py
import pandas as pd
import pickle
import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def evaluate(data_path, model_path):
    df = pd.read_csv(data_path)

    X = df.drop('Loan_Status_Y', axis=1)
    y = df['Loan_Status_Y']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[📊] Model Accuracy: {acc:.4f}")

if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])
>>>>>>> bc29a3b (Final MLOps pipeline with DVC stages and metrics)
