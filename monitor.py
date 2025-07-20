import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping


def generate_reports(full_data_path, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)

    # Load the full dataset
    df = pd.read_csv(full_data_path)

    # Split the data: first 70% as reference, last 30% as current
    split_index = int(len(df) * 0.7)
    reference_data = df.iloc[:split_index].copy()
    current_data = df.iloc[split_index:].copy()

    print("✅ Data loaded and split:")
    print("Reference data shape:", reference_data.shape)
    print("Current data shape:", current_data.shape)

    # Optional: include target if both have it
    column_mapping = ColumnMapping(target="Loan_Status_Y")

    # Generate report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    report.save_html(os.path.join(output_dir, "data_drift_report.html"))

    print("✅ Drift report saved to:", os.path.join(output_dir, "data_drift_report.html"))


if __name__ == "__main__":
    generate_reports("data/clean_train.csv")
