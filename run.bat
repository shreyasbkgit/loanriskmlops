@echo off
CALL conda activate loanmlops

echo [🚀] Starting Loan Risk ML Pipeline...

pip install pandas scikit-learn

python src\preprocess.py data\train.csv data\clean_train.csv
python src\train.py data\clean_train.csv models\model.pkl
python src\evaluate.py data\clean_train.csv models\model.pkl

pause
