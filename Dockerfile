FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
    mosquitto-clients \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY models/ models/

# Set environment variables
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV MQTT_BROKER=localhost
ENV MQTT_PORT=1883
ENV LOGLEVEL=INFO

# Run the application
HEALTHCHECK --interval=5s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os; os.kill(1, 0)" || exit 1

CMD ["python", "-m", "src"]