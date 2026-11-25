"""Read-only check for Flux queries, applied by the gateway before anything reaches InfluxDB."""

from __future__ import annotations

import re

ALLOWED_FLUX_IMPORTS = {"date", "strings", "math", "regexp", "timezone", "influxdata/influxdb/schema"}


def check_readonly_flux(flux: str, bucket: str) -> str | None:
    """Return an error message if a Flux query could write, call out, or read another bucket.

    Flux is a full language: `to()` writes to the DB and packages like `http` or `sql` can send
    data to other servers. Only an allowlist of pure helper packages is accepted. This is a guard,
    not a sandbox: in a real deployment, also give the gateway a token that can only read the bucket.
    """
    code = re.sub(r"//[^\n]*", "", flux)  # ignore comments
    for pkg in re.findall(r'^\s*import\s+(?:\w+\s+)?"([^"]+)"', code, flags=re.M):
        if pkg not in ALLOWED_FLUX_IMPORTS:
            return f'import "{pkg}" is not allowed (allowed: {", ".join(sorted(ALLOWED_FLUX_IMPORTS))})'
    if re.search(r"\b(to|wideTo)\s*\(", code):
        return "writing data (to()) is not allowed: queries must be read-only"
    buckets = re.findall(r'bucket\s*:\s*"([^"]*)"', code)
    if re.search(r"\bbucketID\s*:|\bhost\s*:", code):
        return "only the configured bucket on this server may be queried (no bucketID or host)"
    if not buckets:
        return f'the query must read from(bucket: "{bucket}")'
    if any(b != bucket for b in buckets):
        return f'only bucket "{bucket}" may be queried'
    return None
