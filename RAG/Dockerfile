FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY .env .

# Create directories for data and index
RUN mkdir -p data faiss_index evaluation

# Expose ports
EXPOSE 8000

# Run ingestion then start server
CMD python -m app.ingest && python -m app.main