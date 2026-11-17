# Image for the TSAgent agent (pipeline + LLM loop). InfluxDB and Ollama run in their own containers.
FROM python:3.10-slim

# lightgbm (used by YASA) needs the OpenMP runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tsagent/ tsagent/

# Run as an unprivileged user. Caches go to /tmp because the container's filesystem is read-only.
RUN useradd --uid 1000 --no-create-home agent
USER agent
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba

ENTRYPOINT ["python", "-m", "tsagent"]
CMD ["--help"]
