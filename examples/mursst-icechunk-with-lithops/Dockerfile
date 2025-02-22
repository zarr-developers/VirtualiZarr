# Adapted from https://github.com/lithops-cloud/lithops/blob/master/runtime/aws_lambda/Dockerfile

# Python 3.12
FROM python:3.12-slim

RUN apt-get update \
    # Install aws-lambda-cpp build dependencies
    && apt-get install -y \
    g++ \
    make \
    cmake \
    unzip \
    # cleanup package lists, they are not used anymore in this image
    && rm -rf /var/lib/apt/lists/* \
    && apt-cache search linux-headers-generic

ARG FUNCTION_DIR="/function"

# Copy function code
RUN mkdir -p ${FUNCTION_DIR}

# Update pip
RUN pip install --upgrade --ignore-installed pip wheel six setuptools \
    && pip install --upgrade --no-cache-dir --ignore-installed \
    awslambdaric \
    boto3 \
    redis \
    httplib2 \
    requests \
    numpy \
    scipy \
    pandas \
    pika \
    kafka-python \
    cloudpickle \
    ps-mem \
    tblib \
    psutil

# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Add Lithops
COPY lithops_lambda.zip ${FUNCTION_DIR}
RUN unzip lithops_lambda.zip \
    && rm lithops_lambda.zip \
    && mkdir handler \
    && touch handler/__init__.py \
    && mv entry_point.py handler/

# Put your dependencies here, using RUN pip install... or RUN apt install...
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "handler.entry_point.lambda_handler" ]
