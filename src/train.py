import pandas as pd
<<<<<<< HEAD
=======
import pickle
import yaml
import mlflow
>>>>>>> bc29a3b (Final MLOps pipeline with DVC stages and metrics)
import sys
import joblib
import mlflow
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

<<<<<<< HEAD
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
=======
with open("params.yaml") as f:
    params = yaml.safe_load(f)

test_size = params["train"]["test_size"]
random_state = params["train"]["random_state"]
n_estimators = params["train"]["n_estimators"]

def train(data_path, model_path):
    df = pd.read_csv(data_path)
    X = df.drop('Loan_Status_Y', axis=1)
    y = df['Loan_Status_Y']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=random_state)

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
>>>>>>> bc29a3b (Final MLOps pipeline with DVC stages and metrics)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        mlflow.sklearn.log_model(model, "model")
        print(f"[✅] Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_model(sys.argv[1], sys.argv[2])
