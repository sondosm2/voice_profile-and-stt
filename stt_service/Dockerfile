FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
# Install torch/torchaudio matching the CUDA build first (big layer, changes rarely)
RUN pip install --upgrade pip 
COPY requirements.txt .
RUN pip install -r requirements.txt



COPY . .

RUN mkdir -p /app/uploads

CMD ["python3","main.py"]