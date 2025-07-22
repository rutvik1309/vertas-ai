FROM python:3.10-slim

# Install system dependencies for newspaper3k and others
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5005

CMD ["python", "app.py"]
