FROM ubuntu:18.04

ENV TZ=Europe/Berlin

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install -y python3 python3-pip python3-dev python3-opencv build-essential tesseract-ocr  
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python pytesseract tensorflow flask Pillow

WORKDIR /usr/src/app
COPY . .

RUN rm lanms/adaptor.so

EXPOSE 5000

CMD [ "python3", "./app.py" ]