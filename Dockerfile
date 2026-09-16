FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-serving.txt


FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /install /usr/local

COPY src/ src/
COPY configs/ configs/

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
