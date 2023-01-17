FROM python:3.8-alpine

WORKDIR /usr/src/app

RUN mkdir data
COPY data/classifier.pkl ./data/
COPY data/extractor.pkl ./data/
COPY src ./
COPY requirements.txt ./

RUN apk update && apk add python3-dev gcc g++ libc-dev libffi-dev

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# Excpose the port to serve the HTTP requests
EXPOSE 8080

CMD ["python", "./src/service/flask_app.py"]