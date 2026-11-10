# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# System dependencies for OpenCV / Pillow / reportlab
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit dashboard
CMD ["python", "run.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
