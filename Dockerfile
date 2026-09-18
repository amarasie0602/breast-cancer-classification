FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir --only-binary :all: --prefix=/install -r requirements-serving.txt


FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /install /usr/local

COPY src/ src/
COPY configs/ configs/

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
