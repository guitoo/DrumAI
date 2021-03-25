FROM python:3.7-slim-buster

RUN apt-get -y update
RUN apt-get -y install libsndfile1 ffmpeg

COPY ./requirements_api.txt /requirements.txt
RUN pip install -r /requirements.txt

