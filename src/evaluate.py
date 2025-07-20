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
