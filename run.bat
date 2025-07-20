@echo off
CALL conda activate loanmlops

<<<<<<< HEAD
echo [🚀] Starting Loan Risk ML Pipeline...

pip install pandas scikit-learn

=======
=======
>>>>>>> bc29a3b (Final MLOps pipeline with DVC stages and metrics)
echo [🚀] Starting Loan Risk ML Pipeline...

pip install pandas scikit-learn

<<<<<<< HEAD
REM Step 2: Run scripts
>>>>>>> e7475f2 (dvc)
=======
>>>>>>> bc29a3b (Final MLOps pipeline with DVC stages and metrics)
python src\preprocess.py data\train.csv data\clean_train.csv
python src\train.py data\clean_train.csv models\model.pkl
python src\evaluate.py data\clean_train.csv models\model.pkl

pause
