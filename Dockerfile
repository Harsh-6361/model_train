FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p artifacts/models artifacts/logs data

# Expose port for API
EXPOSE 8000

# Default command
CMD ["python", "scripts/serve.py", "--port", "8000", "--host", "0.0.0.0"]
