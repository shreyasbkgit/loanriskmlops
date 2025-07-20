import pandas as pd
import sys
import joblib
import mlflow
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def train_model(data_path, model_path):
    params = load_params()["train"]

    df = pd.read_csv(data_path)
    X = df.drop("Loan_Status_Y", axis=1)
    y = df["Loan_Status_Y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"]
    )

    clf = RandomForestClassifier(n_estimators=params["n_estimators"], random_state=params["random_state"])
    clf.fit(X_train, y_train)

    joblib.dump(clf, model_path)

    mlflow.set_experiment("LoanApproval")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.sklearn.log_model(clf, "loan_model")

    print(f"[✅] Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_model(sys.argv[1], sys.argv[2])
