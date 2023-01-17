FROM python:3.8

WORKDIR /usr/src/app

COPY data/classifier.pkl ./data/classifier.pkl
COPY data/extractor.pkl ./data/extractor.pkl
ADD  src/ ./src
COPY requirements.txt ./requirements.txt

RUN apt-get update 
RUN apt-get install build-essential -y
RUN apt-get install python3-dev -y
RUN apt-get install gfortran -y
RUN apt-get install make -y
RUN apt-get install libopenblas-dev -y
RUN apt-get install cmake -y

RUN pip install --no-cache-dir --upgrade pip 
RUN pip install --no-cache-dir -r requirements.txt

# Excpose the port to serve the HTTP requests
EXPOSE 8080

CMD ["python", "./src/service/flask_app.py"]