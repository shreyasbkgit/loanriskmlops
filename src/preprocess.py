# src/preprocess.py
import pandas as pd
import os
import sys

def preprocess(input_path, output_path):
    print(f"[ℹ️] Reading from: {input_path}")
    print(f"[ℹ️] Will save to: {output_path}")

    df = pd.read_csv(input_path)

    df.drop(['Loan_ID'], axis=1, inplace=True)

    # Fill missing values
    df['Gender'] = df['Gender'].fillna('Male')
    df['Married'] = df['Married'].fillna('Yes')
    df['Dependents'] = df['Dependents'].fillna('0')
    df['Self_Employed'] = df['Self_Employed'].fillna('No')
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(1.0)

    # Encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    # ✅ Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ✅ Save output
    df.to_csv(output_path, index=False)
    print(f"[✅] Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess(sys.argv[1], sys.argv[2])
