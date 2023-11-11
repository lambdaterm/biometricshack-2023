FROM python:3.10.6

WORKDIR /app

ADD data /app/data

COPY requirements.txt /app/

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && \
    python3 -m pip install -U pip && python3 -m pip install -r requirements.txt

COPY model.onnx run.py /app/

CMD python3 run.py
