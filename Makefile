include .env
export

fetch_dataset:
	python -m pipeline_model.fetch_kaggle_dataset

train: fetch_dataset
	python -m model_training