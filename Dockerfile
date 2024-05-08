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
    sklego \
    fairlearn \
    fairgbm \
    tqdm \
    lightgbm \
    folktables

# Install jupyter
RUN pip install jupyter \
    jupyterlab  \
    notebook

# docker build -t m2fgb:$USER -f Dockerfile --build-arg OUTSIDE_GROUP=`/usr/bin/id -ng $USER` --build-arg OUTSIDE_GID=`/usr/bin/id -g $USER` --build-arg OUTSIDE_USER=$USER --build-arg OUTSIDE_UID=$UID .

# Without jupyter:
# docker run -it --userns=host --name m2fgb -v /work/$USER:/work/$USER m2fgb:$USER  /bin/bash

# With jupyter:
# docker run -it --userns=host --name m2fgb -v /work/$USER:/work/$USER -p 30001:30001 m2fgb:$USER  /bin/bash

# To enter the container as non-root:
# docker exec -ti -u $USER m2fgb bash

# To run jupyter:
# docker exec -ti -u $USER m2fgb bash -c "jupyter-lab --port 30001 --ip 0.0.0.0"

