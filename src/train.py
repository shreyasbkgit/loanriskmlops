import pandas as pd
import pickle
import yaml
import mlflow
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        mlflow.sklearn.log_model(model, "model")
        print(f"[✅] Model trained and saved to {model_path}")

if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
