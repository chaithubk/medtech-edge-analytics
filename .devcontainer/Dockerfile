FROM python:3.10-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    vim \
    nano \
    curl \
    wget \
    tree \
    htop \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    mosquitto \
    mosquitto-clients \
    openssh-client \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# Set environment variables
ENV MQTT_BROKER=localhost
ENV MQTT_PORT=1883
ENV LOGLEVEL=INFO
ENV PYTHONPATH=/workspace/src:$PYTHONPATH

CMD ["/bin/bash"]