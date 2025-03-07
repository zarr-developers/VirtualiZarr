# Use AWS Lambda base image for Python 3.11
FROM public.ecr.aws/lambda/python:3.11

ARG FUNCTION_DIR

# Set working directory
WORKDIR /var/task

# Update system libraries and install necessary utilities
RUN yum update -y && \
    yum install -y wget unzip tar gzip git && \
    yum clean all

# Install uv package manager and move it to /usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv ~/.local/bin/uv /usr/local/bin/uv && \
    chmod +x /usr/local/bin/uv

# Verify uv installation
RUN uv --version

RUN uv pip install --upgrade pip wheel six setuptools --system \
    && uv pip install --upgrade --no-cache-dir --system \
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

# Set environment variables for Lambda
ENV PYTHONPATH="/var/lang/lib/python3.11/site-packages:${FUNCTION_DIR}"

# Copy and install dependencies from requirements.txt using uv
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --no-cache-dir -r /tmp/requirements.txt --system

# Copy application code
COPY lithops_lambda.zip ${FUNCTION_DIR}
RUN unzip lithops_lambda.zip \
    && rm lithops_lambda.zip \
    && mkdir handler \
    && touch handler/__init__.py \
    && mv entry_point.py handler/

# Set Lambda entry point
CMD [ "handler.entry_point.lambda_handler" ]
