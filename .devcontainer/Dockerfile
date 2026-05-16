FROM python:3.10-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies including Java 17 for Synthea
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    vim \
    openjdk-17-jre \
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
    bubblewrap \
    socat \
    shellcheck \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install actionlint
RUN curl -fsSL "https://github.com/rhysd/actionlint/releases/download/v1.7.7/actionlint_1.7.7_linux_amd64.tar.gz" | tar -xz -C /usr/local/bin actionlint

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

WORKDIR /workspaces/medtech-edge-analytics

# Copy and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt -r requirements-dev.txt

# Set environment variables
ENV MQTT_BROKER=localhost
ENV MQTT_PORT=1883
ENV LOGLEVEL=INFO
ENV PYTHONPATH=/workspaces/medtech-edge-analytics/src:$PYTHONPATH

CMD ["/bin/bash"]