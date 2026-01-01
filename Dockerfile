FROM python:3.11-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create data directory for persistent log
RUN mkdir -p /data

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY processor.py .
COPY server.py .

# Default log path (can be overridden via env)
ENV LOG_PATH=/data/processing_log.html

# Run server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]