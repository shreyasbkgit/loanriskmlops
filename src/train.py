# src/train.py
import pandas as pd
import pickle
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train(data_path, model_path):
    df = pd.read_csv(data_path)

    X = df.drop('Loan_Status_Y', axis=1)
    y = df['Loan_Status_Y']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"[✅] Model trained and saved to {model_path}")

if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
