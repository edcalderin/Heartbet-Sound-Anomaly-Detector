include .env
export

fetch-dataset:
	python -m backend_app.fetch_kaggle_dataset