FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      software-properties-common \
      curl \
      git \
      tmux \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3.11 \
      python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/pip3 /usr/local/bin/pip

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    MPLBACKEND=Agg

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

WORKDIR /workspace
COPY . /workspace/

RUN pip install .

ENTRYPOINT ["python3", "-m", "matrix_completion.cli"]
CMD ["--help"]
