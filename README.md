# Heartbet Sound Anomaly Detector

*Machine Learning Zoomcap Capstone 2 project*

## Table of Contents

<!--ts-->
* [Problem statement](#problem-statement)
* [Directory layout](#directory-layout)
* [Setup](#setup)
* [Training model](#training-model)
* [Notebooks](#notebooks)
* [Running the app with Docker (Recommended)](#running-the-app-with-docker-recommended)
    * [Streamlit UI](#streamlit-ui)
    * [Backend service](#backend-service)
    * [Testing the app](#testing-the-app)
* [Running with Kubernetes](#running-with-kubernetes)
* [Application running on Cloud](#application-running-on-cloud)
* [Architecture](#architecture)
* [Checkpoints](#checkpoints)
* [References](#references)
<!--te-->

## Problem statement

This project addresses the critical challenge of early detection of heart anomalies by providing individuals with a reliable tool to assess their cardiovascular health based on heart sound analysis. Whether users are proactively monitoring their well-being or expressing concerns about potential cardiac conditions, an accurate anomaly detector is essential. To meet this need, a robust deep learning model has been developed and trained using the "Heartbeat Sound Anomaly Detection Database" [(See on references)](#references), a comprehensive collection of audio files capturing diverse heart sounds.

The database encompasses information on ten crucial audio features, including variations in heartbeats, murmurs, tones, and other distinctive sound patterns associated with cardiac health. These audio variables were meticulously analyzed to uncover hidden patterns and gain valuable insights into factors influencing heart anomalies. Subsequently, a cutting-edge deep learning model was rigorously trained, validated, and deployed in real-time, enabling efficient and accurate detection of heart anomalies through the analysis of heartbeat sounds.

By leveraging this innovative solution, individuals can proactively monitor their cardiovascular health, receive timely alerts for potential anomalies detected in heartbeat sounds, and take preventive measures to ensure their well-being. The implementation of advanced technology in this domain empowers users to make informed decisions about their heart health, contributing to early intervention and improved cardiovascular outcomes.

## Directory layout

```
.
├── backend_app             # Backend files
├── config_management       # Config files
├── frontend_app            # Directory with files to create Streamlit UI application
├── images                  # Assets
├── model_pipeline          # Files to preprocess and train the model
|── pth_models              # Trained models
└── notebooks               # Notebooks used to explore data and select the best model

7 directories
```

## Setup

1. Rename `.env.example` to `.env` and set your Kaggle credentials in this file.
2. Sign into [Kaggle account](https://www.kaggle.com).
3. Go to https://www.kaggle.com/settings
4. Click on `Create new Token` to download the `kaggle.json` file
5. Copy `username` and `key` values and past them into `.env` variables respectively.
6. Make installation:

<!--ts-->
* For UNIX-based systems and Windows (WSL), you do not need to install make.
* For Windows without WSL:
    * Install chocolatey from [here](https://chocolatey.org/install)
    * Then, `choco install make`.
<!--te-->

## Training model

This project already provides a trained model in `/pth_models` directory to deploy the application, but if you wish generate new models, then follow the next steps.

Take into account it will take several minutes even with an available gpu:

* Create a conda environment: `conda create -n <name-of-env> python=3.11`
* Start environment: `conda activate <name-of-env>` or `source activate <name-of-env>`

* Install pytorch dependencies:  

__With GPU__:  
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`  

__With CPU__:  
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`  

* Install rest of dependencies:  
`pip install kaggle pandas numpy seaborn pyyaml numpy matplotlib ipykernel librosa`
* Run `make train`
* The models will be saved to `/pth_models` directory with the following pattern:
`epoch_{epoch}_acc={accuracy in training}_val_acc={accuracy in validation}.pth`

You will see different models per epoch, you must choose one of them afterwards and set it in the configuration file. The `model_name` property:

```yaml
title: 'Hearbet Sound Anomaly Detector API'

unzipped_directory: unzipped_data
kaggle_dataset: kinguistics/heartbeat-sounds

model_name: pth_models/epoch_4_acc=0.8416_val_acc=0.7614.pth ## Replace for the new model

...
```
You are free to consciously manipulate this configuration file with different parameters.

## Notebooks

Run notebooks in `notebooks/` directory to conduct Exploratory Data Analysis and model training. The environment created in previous section can be also used here.

## Running the app with Docker (Recommended)

Run `docker-compose up --build` to start the services at first time or `docker-compose up` to start services after the initial build

* `http://localhost:8501` (Streamlit UI)
* `http://localhost:8000` (Backend service)

The output should look like this:

![Alt text](./images/docker-output.png)

### Streamlit UI

User interface designed using Streamlit to interact with backend endpoints:

![Alt text](./images/streamlit_app.png)

### Testing the app

If you did not train any model, then run `make fetch_dataset` to download the dataset from Kaggle website and test the prediction endpoint.

* Stop the services with `docker-compose down`

## Running with Kubernetes

Assuming you have `kind` tool installed on your system, follow the instructions on [kube_instructions.md](instructions/kube_instructions.md)

## Application running on Cloud

![Alt text](./images/awseb.png)

The application has been deployed to cloud using AWS ElasticBeanstalk, both frontend and backend were separately deployed using `eb` command:

## Architecture

![Alt text](./images/architecture.png)

## Checkpoints

- [x] Problem description
- [x] EDA
- [x] Model training
- [x] Exporting notebook to script
- [x] Reproducibility
- [x] Model deployment
- [x] Dependency and enviroment management
- [x] Containerization
- [x] Cloud deployment
- [ ] Linter
- [ ] CI/CD workflow
- [ ] Pipeline orchestration
- [ ] Unit tests

## ✉️ Contact
**LinkedIn:** https://www.linkedin.com/in/erick-calderin-5bb6963b/  
**e-mail:** edcm.erick@gmail.com

## Enjoyed this content?
Explore more of my work on [Medium](https://medium.com/@erickcalderin) 

I regularly share insights, tutorials, and reflections on tech, AI, and more. Your feedback and thoughts are always welcome!

## References

* [Dataset] https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds
* https://pytorch.org/audio
* https://medium.com/@muhammad2000ammar/mastering-transfer-learning-with-pytorch-d1521f3a6a6e
