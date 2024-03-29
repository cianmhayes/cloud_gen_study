FROM ubuntu

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y python3-pip

COPY . .

RUN pip3 install -r linux_requirements.txt