FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (for Tesseract OCR and PDF processing)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ara \
    poppler-utils \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# --- The Smart Trick Here ---
# Install CPU-only version of PyTorch (small size ~150MB instead of the standard 2GB)
# This command precedes requirements.txt to prevent version conflicts
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Keep the container running in the background
CMD ["tail", "-f", "/dev/null"]