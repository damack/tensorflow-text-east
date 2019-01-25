FROM ubuntu:16.04

RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install -y python3 python3-pip python3-dev build-essential tesseract-ocr mpich
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python pytesseract flask Pillow lanms
RUN pip3 --no-cache-dir install https://github.com/mind/wheels/releases/download/tf1.7-cpu/tensorflow-1.7.0-cp35-cp35m-linux_x86_64.whl
RUN apt-get install -y libsm6

WORKDIR /usr/src/app
COPY . .

EXPOSE 5000

CMD [ "python3", "./app.py" ]