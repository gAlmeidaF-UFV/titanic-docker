FROM python:3.8-slim-buster

RUN apt-get update\
    && mkdir /code\
    && mkdir /datasets\
    && mkdir /output

COPY ./code /code

COPY ./dataset /dataset

RUN pip install --upgrade pip\
    && pip install -r /code/requirements.txt

ENTRYPOINT [ "python3" ]
CMD ["/code/titanic.py"]