include .env
export

fetch_dataset:
	python -m model_training.fetch_kaggle_dataset

train: fetch_dataset
	python -m model_training