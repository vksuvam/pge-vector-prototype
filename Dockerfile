FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/images qdrant_storage

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')" || exit 1

# Run uvicorn on port 7860 (required for HF Spaces)
# HF Spaces expects the app to listen on 0.0.0.0:7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]