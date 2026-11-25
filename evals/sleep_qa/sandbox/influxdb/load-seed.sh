#!/bin/bash
# Runs once, after InfluxDB's setup mode has created the org, bucket and admin token.
set -euo pipefail
influx write --org "$DOCKER_INFLUXDB_INIT_ORG" --bucket "$DOCKER_INFLUXDB_INIT_BUCKET" \
  --precision ns --file /seed/seed.lp
touch /var/lib/influxdb2/.seeded     # the healthcheck waits for this
