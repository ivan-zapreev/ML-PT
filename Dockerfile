FROM python:3.8-alpine

WORKDIR /usr/src/app

RUN mkdir data
COPY data/classifier.pkl ./data/
COPY data/extractor.pkl ./data/
COPY src ./
COPY requirements.txt ./

RUN apk update
RUN apk add --no-cache python3-dev
RUN apk add --no-cache build-base
RUN apk add --no-cache gfortran 
RUN apk add --no-cache make
RUN apk add --no-cache openblas-dev
RUN apk add --no-cache cmake

RUN pip install --no-cache-dir --upgrade pip 
RUN pip install --no-cache-dir -r requirements.txt

# Excpose the port to serve the HTTP requests
EXPOSE 8080

CMD ["python", "./src/service/flask_app.py"]