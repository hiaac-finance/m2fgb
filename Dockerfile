FROM ubuntu:22.04 
# nvidia/cuda:12.2.0-runtime-ubuntu22.04

ARG OUTSIDE_GID
ARG OUTSIDE_UID
ARG OUTSIDE_USER
ARG OUTSIDE_GROUP

RUN groupadd --gid ${OUTSIDE_GID} ${OUTSIDE_GROUP}
RUN useradd --create-home --uid $OUTSIDE_UID --gid $OUTSIDE_GID $OUTSIDE_USER

ENV SHELL=/bin/bash

WORKDIR /work/

# Build with some basic utilities
RUN apt-get update 

RUN apt-get install -y \
    python3-pip \
    apt-utils \
    vim \
    git \
    unzip 

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install python packages
RUN pip3 install --upgrade pip

# Install python packages
RUN pip install numpy \
    matplotlib \
    pandas \
    scikit-learn \
    xgboost \
    optuna \
    gdown \ 
    fairlearn \
    fairgbm \
    tqdm \
    lightgbm


