FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV API_URL=http://api:8000

EXPOSE 8000 8050

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
