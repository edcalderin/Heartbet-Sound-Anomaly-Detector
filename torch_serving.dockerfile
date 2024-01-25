FROM pytorch/torchserve:latest-cpu

USER root

RUN apt-get update && apt-get --assume-yes install zip

WORKDIR /home/model-server/

ENV BUILD_DIR=/home/model-server

COPY ["copy_artifacts.sh", \
    "pth_models/epoch_4_acc=0.8416_val_acc=0.7614.pth", \
    "model_pipeline/pretrained_network.py", \
    "model_pipeline/preprocessing.py", \
    "model_pipeline/model_handler.py", \
    "model_pipeline/ts_requirements.txt", \
    "/home/model-server/"]

RUN bash /home/model-server/copy_artifacts.sh

RUN torch-model-archiver \
    --export-path /home/model-server/model-store \
    --model-name hearbet-anomaly-detector \
    --version 1.0 \
    --requirements-file /home/model-server/ts_requirements.txt \
    --serialized-file /home/model-server/epoch_4_acc=0.8416_val_acc=0.7614.pth \
    --model-file /home/model-server/pretrained_network.py \
    --handler /home/model-server/model_handler.py \
    --extra-files /home/model-server/model_pipeline.zip

RUN printf "\ninstall_py_dep_per_model=true" >> /home/model-server/config.properties
RUN printf "\ncpu_launcher_enable=true" >> /home/model-server/config.properties
RUN printf "\ncpu_launcher_args=--use_logical_core" >> /home/model-server/config.properties

USER model-server

CMD ["torchserve", \
    "--start", \
    "--model-store", \
    "/home/model-server/model-store", \
    "--models", \
    "hearbet_model=hearbet-anomaly-detector.mar"]
