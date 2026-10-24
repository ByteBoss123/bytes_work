# Scoring service image for Cloud Run.
# The model is not committed to Git, so the image trains it from the seed-42
# synthetic simulator at build time. The model and its monitoring reference are
# written together, so the service's fingerprint check passes.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pbm ./pbm
COPY sql ./sql
RUN python -m pbm.pipeline --n 60000 --seed 42 --out reports \
    && rm -f reports/claims.sqlite reports/test_predictions.csv

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

# Cloud Run sets PORT. Telemetry lives in /tmp: it is per-instance and resets
# when the instance is replaced (see deploy/README.md, Limits).
CMD ["sh", "-c", "exec python -m pbm.service --host 0.0.0.0 --port ${PORT:-8080} --telemetry /tmp/telemetry.sqlite"]
