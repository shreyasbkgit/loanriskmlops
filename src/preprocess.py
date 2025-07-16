# src/preprocess.py
import pandas as pd
import sys

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)

    df.drop(['Loan_ID'], axis=1, inplace=True)

    # Fill missing values
    df.loc[:, 'Gender'] = df['Gender'].fillna('Male')
    df.loc[:, 'Married'] = df['Married'].fillna('Yes')
    df.loc[:, 'Dependents'] = df['Dependents'].fillna('0')
    df.loc[:, 'Self_Employed'] = df['Self_Employed'].fillna('No')
    df.loc[:, 'LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df.loc[:, 'Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df.loc[:, 'Credit_History'] = df['Credit_History'].fillna(1.0)

    # Encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    df.to_csv(output_path, index=False)
    print(f"[✅] Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess(sys.argv[1], sys.argv[2])
