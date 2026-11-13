# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# CPU-only PyTorch keeps the image a fraction of the size of the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

FROM python:3.12-slim-bookworm
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    GNNFRAUD_DATA_DIR=/app/data \
    GNNFRAUD_ARTIFACTS_DIR=/app/artifacts \
    GNNFRAUD_RESULTS_DIR=/app/results
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
RUN mkdir -p data artifacts results && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=120s --retries=3 \
  CMD python -c "import json,sys,urllib.request; sys.exit(json.load(urllib.request.urlopen('http://127.0.0.1:8000/healthz'))['status'] != 'ok')"
CMD ["uvicorn", "gnn_fraud.api:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
