FROM python:3.11-slim

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy app files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
