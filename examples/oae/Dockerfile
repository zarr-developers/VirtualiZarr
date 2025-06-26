FROM python:3.12-slim-bookworm

# Install system dependencies including git
RUN apt-get update \
    # Install aws-lambda-cpp build dependencies
    && apt-get install -y \
      g++ \
      make \
      cmake \
      unzip \
      git \
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

# Install packages from specific branches
# TODO all of these could be replaced with `pip install 'virtualizarr[hdf]'` once I can use a released version
RUN pip install \
    numpy>=2.0.0 \
    universal-pathlib \
    ujson \
    packaging \
    zarr>=3.0.2 \
    numcodecs>=0.16.1 \
    icechunk \
    s3fs \
    fsspec \
    requests \
    aiohttp \
    h5py \
    hdf5plugin \
    imagecodecs \
    imagecodecs-numcodecs==2024.6.1 \
    obstore>=0.5.1 \
    git+https://github.com/zarr-developers/VirtualiZarr.git@develop

# Entry point configuration for AWS Lambda
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "handler.entry_point.lambda_handler" ]
