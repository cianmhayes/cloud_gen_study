FROM nvidia/cuda:9.0-base-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y python3-pip
RUN apt-get install -y libssl-dev

COPY . .

RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html