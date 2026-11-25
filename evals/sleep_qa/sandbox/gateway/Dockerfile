# Query gateway: standard library only. Build context: this directory.
FROM python:3.12-alpine
WORKDIR /app
COPY flux_guard.py gateway.py ./
RUN mkdir /logs && chown 1000 /logs
USER 1000
CMD ["python", "-u", "gateway.py"]
