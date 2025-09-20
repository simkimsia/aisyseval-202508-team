# Lightweight Python base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py /app/main.py
COPY config.txt /app/config.txt

# Expose port
EXPOSE 8000

# Start the app (PORT can be set in config.txt or via env)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
