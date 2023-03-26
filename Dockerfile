FROM anibali/pytorch:2.0.0-cuda11.8
USER root

WORKDIR /src

RUN sudo apt-get update
RUN sudo apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN sudo apt-get --allow-releaseinfo-change update && \
    sudo apt-get install -y --no-install-recommends \
    curl \
    sudo \
    vim

RUN pip install matplotlib \
    seaborn \
    lightning \
    comet_ml \
    wandb \
    sklearn \
    boto3 \
    tenacity \
    pandas \
    plotly \
    scipy \
    torchmetrics

COPY .. .