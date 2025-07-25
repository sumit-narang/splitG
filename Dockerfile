FROM python:3.12-slim

# Install Tesseract OCR system package
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy everything
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Railway uses
EXPOSE 8080

# Start the server with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
