# Scoring service image for Cloud Run.
# Serves the model trained by CI on real CMS Part D formulary files. The service checks the
# model's SHA-256 against monitoring_reference.json before it starts.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pbm ./pbm
COPY reports/model.joblib reports/monitoring_reference.json ./reports/

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

# Cloud Run sets PORT. Telemetry lives in /tmp: it is per-instance and resets
# when the instance is replaced (see deploy/README.md, Limits).
CMD ["sh", "-c", "exec python -m pbm.service --host 0.0.0.0 --port ${PORT:-8080} --telemetry /tmp/telemetry.sqlite"]
