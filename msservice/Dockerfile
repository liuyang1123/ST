FROM python:2-wheezy

ADD requirements.txt /app/requirements.txt
WORKDIR /app/
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
RUN pip install numpy
RUN pip install pomegranate==0.4.0

RUN adduser --disabled-password --gecos '' myuser
