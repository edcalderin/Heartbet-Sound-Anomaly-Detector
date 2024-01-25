FROM python:3.10-slim-buster

WORKDIR /backend_app_hearbet_detector

COPY backend_app/requirements.txt backend_app/

RUN pip install --upgrade pip

RUN pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install fastapi==0.109.0 uvicorn==0.27.0 pyyaml==6.0.1 timm==0.9.12 pysoundfile==0.9.0.post1 python-multipart==0.0.6
RUN apt-get update -y && apt-get -y install libsox-dev

COPY backend_app backend_app/
COPY model_pipeline/pretrained_network.py model_pipeline/
COPY config_management config_management/
COPY pth_models pth_models/

EXPOSE 8000

ENTRYPOINT [ "uvicorn", "--host=0.0.0.0", "--port=8000", "backend_app.__main__:app" ]