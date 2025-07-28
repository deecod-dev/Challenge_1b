# Use python base image compatible with AMD64 architecture
FROM --platform=linux/amd64 python:3.10-slim

# Avoid writing .pyc files and force stdout/stderr flushing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required by PyMuPDF and pdfplumber
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    poppler-utils \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model for offline use
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY main.py .

# Set entrypoint for running the analyzer
ENTRYPOINT ["python", "main.py"]
