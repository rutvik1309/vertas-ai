FROM python:3.10-slim

# Prevent .pyc files and ensure logs are printed directly
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by your Python packages
RUN apt-get update && apt-get install -y \
    python3-dev \
    libxml2-dev \
    libxslt-dev \
    libjpeg-dev \
    zlib1g-dev \
    libffi-dev \
    libssl-dev \
    build-essential \
    poppler-utils \
    gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY requirements.txt .
COPY nltk_setup.py .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download NLTK data
RUN python nltk_setup.py

# Now copy the rest of the project files
COPY . .

# Expose the port the app runs on
EXPOSE 5005

# Run the app
CMD ["python", "app.py"]
