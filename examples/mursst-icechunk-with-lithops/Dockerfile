FROM public.ecr.aws/lambda/python:3.11

ARG FUNCTION_DIR

# Update libs
RUN yum update -y && \
    yum install -y wget unzip && \
    yum clean all

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda && \
    rm miniconda.sh

COPY lithops-env.yml /tmp/lithops-conda.yml

# Create conda environment
RUN /opt/miniconda/bin/conda update -n base -c defaults conda && \
    /opt/miniconda/bin/conda env create --file /tmp/lithops-conda.yml --prefix /opt/conda-env

# Install Lithops dependencies
RUN /opt/conda-env/bin/pip install --upgrade --no-cache-dir --ignore-installed \
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

# Replace python interpreter with conda's
RUN mv /var/lang/bin/python3.11 /var/lang/bin/python3.11-clean && \
    ln -sf /opt/conda-env/bin/python /var/lang/bin/python3.11

ENV PYTHONPATH="/var/lang/lib/python3.11/site-packages:${FUNCTION_DIR}"

ENV PATH="${PATH}:/opt/conda-env/bin/"

# Install lithops
COPY lithops_lambda.zip ${FUNCTION_DIR}
RUN unzip lithops_lambda.zip \
    && rm lithops_lambda.zip \
    && mkdir handler \
    && touch handler/__init__.py \
    && mv entry_point.py handler/

CMD [ "handler.entry_point.lambda_handler" ]
