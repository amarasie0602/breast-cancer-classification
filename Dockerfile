FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements-serving.lock.txt .
RUN pip install --no-cache-dir --require-hashes --only-binary :all: --prefix=/install -r requirements-serving.lock.txt


FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /install /usr/local

COPY src/ src/
COPY configs/ configs/
COPY checkpoints/best_mag40.pt checkpoints/best_mag40.pt
COPY checkpoints/best_subtype.pt checkpoints/best_subtype.pt

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
