# InfluxDB with the eval scenario baked in. Build context: the repository root.
#
# Stage 1 runs evals/sleep_qa/make_seed.py to write the scenario as line protocol.
# Stage 2 is stock InfluxDB; on first start its setup mode creates the org/bucket/token and then
# runs /docker-entrypoint-initdb.d/*.sh, which loads the seed file.

FROM python:3.12-slim AS seed
WORKDIR /src
RUN pip install --no-cache-dir numpy pandas
COPY dataloader/__init__.py dataloader/pipeline.py dataloader/synthetic.py dataloader/
COPY evals/sleep_qa/scenario.py evals/sleep_qa/make_seed.py evals/sleep_qa/
RUN python evals/sleep_qa/make_seed.py /seed.lp

FROM influxdb:2.7
COPY --from=seed /seed.lp /seed/seed.lp
COPY evals/sleep_qa/sandbox/influxdb/load-seed.sh /docker-entrypoint-initdb.d/load-seed.sh
