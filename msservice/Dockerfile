FROM gcr.io/tensorflow/tensorflow

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /code
WORKDIR /code

ADD . /code/

ADD requirements.txt /code/
RUN pip install -r requirements.txt
# RUN pip install pomegranate --no-cache-dir
RUN pip install https://github.com/gnzlo789/pomegranate/archive/master.zip
