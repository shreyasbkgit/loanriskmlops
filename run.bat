@echo off
CALL conda activate loanmlops

echo [🚀] Starting Loan Risk ML Pipeline...

pip install pandas scikit-learn

=======
echo [🚀] Starting Loan Risk ML Pipeline...


REM Step 1: Install dependencies globally (not recommended long-term)
pip install pandas scikit-learn

REM Step 2: Run scripts
>>>>>>> e7475f2 (dvc)
python src\preprocess.py data\train.csv data\clean_train.csv
python src\train.py data\clean_train.csv models\model.pkl
python src\evaluate.py data\clean_train.csv models\model.pkl

pause
