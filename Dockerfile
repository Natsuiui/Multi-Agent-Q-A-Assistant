FROM python:3.10-slim

WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libnss3 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app files
COPY . .

# Download the model
RUN python download_model.py

# Expose port 80 for Streamlit
EXPOSE 80

CMD ["streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]
