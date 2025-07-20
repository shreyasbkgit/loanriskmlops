dvc stage add -n evaluate -d src/evaluate.py -d data/clean_train.csv -d models/model.pkl -o metrics.json -M metrics.json python src/evaluate.py data/clean_train.csv models/model.pkl metrics.json
python src/evaluate.py data/clean_train.csv models/model.pkl metrics.json
