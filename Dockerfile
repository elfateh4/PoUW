# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install PoUW package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 pouw && \
    chown -R pouw:pouw /app
USER pouw

# Expose ports
EXPOSE 8000 8080 8545 8546

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pouw; print('PoUW is healthy')" || exit 1

# Default command
CMD ["python", "-m", "pouw.node", "--help"]
