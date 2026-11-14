FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-checks.txt ./
RUN pip install --no-cache-dir -r requirements-checks.txt
COPY . .
RUN scripts/fetch_weights.sh
ENV GENEVAL_DSN=postgresql://geneval:geneval@postgres:5432/geneval \
    GENEVAL_REDIS=redis://redis:6379/0 \
    GENEVAL_ARTIFACTS=/data/artifacts
CMD ["uvicorn", "geneval.api:app", "--host", "0.0.0.0", "--port", "8000"]
