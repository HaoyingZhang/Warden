"""Query gateway: the only way from the agent's container to InfluxDB.

POST /query?max_rows=N   body: a Flux query (text)   ->   JSON {row_count, truncated, rows} or {error}

- Holds the InfluxDB token; the agent never sees it, nor the InfluxDB address.
- Accepts read-only Flux on one bucket only (flux_guard.py).
- Pins Flux's now() to the scenario date, so relative ranges like range(start: -7d) behave as if
  today were the user's "today".
- Appends every request to /logs/queries.jsonl; the eval's scorer reads that file.
Standard library only.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from flux_guard import check_readonly_flux

INFLUX_URL = os.environ["INFLUX_URL"].rstrip("/")
INFLUX_TOKEN = os.environ["INFLUX_TOKEN"]
INFLUX_ORG = os.environ["INFLUX_ORG"]
BUCKET = os.environ["INFLUX_BUCKET"]
FLUX_NOW = os.environ.get("FLUX_NOW")             # e.g. 2026-11-17T10:00:00Z
LOG_PATH = os.environ.get("LOG_PATH", "/logs/queries.jsonl")
MAX_ROWS_CAP = 1000
_lock = threading.Lock()


def pin_now(flux: str) -> str:
    """Insert `option now = ...` after the import lines (Flux requires imports first)."""
    if not FLUX_NOW or re.search(r"^\s*option\s+now\b", flux, flags=re.M):
        return flux
    lines = flux.splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith(("import ", "//"))):
        i += 1
    return "\n".join(lines[:i] + [f"option now = () => {FLUX_NOW}"] + lines[i:])


def run_query(flux: str, max_rows: int) -> dict:
    body = json.dumps({"query": pin_now(flux), "type": "flux",
                       "dialect": {"header": True, "annotations": [], "delimiter": ","}}).encode()
    req = urllib.request.Request(
        f"{INFLUX_URL}/api/v2/query?org={INFLUX_ORG}", data=body, method="POST",
        headers={"Authorization": f"Token {INFLUX_TOKEN}", "Content-Type": "application/json",
                 "Accept": "application/csv"})
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("message", "")
        except ValueError:
            msg = str(e)
        return {"error": f"InfluxDB returned HTTP {e.code}: {msg[:1000]}"}
    rows, header, truncated = [], None, False
    with resp:
        for raw in io.TextIOWrapper(resp, encoding="utf-8"):
            line = raw.rstrip("\r\n")
            if not line.strip():                  # blank line: next table, new header
                header = None
                continue
            values = next(csv.reader([line]))
            if header is None:
                header = values
                continue
            if len(rows) >= max_rows:
                truncated = True
                break
            rows.append({k: v for k, v in zip(header, values) if k not in ("", "result", "table")})
    out = {"row_count": len(rows), "truncated": truncated, "rows": rows}
    if truncated:
        out["note"] = f"Only the first {len(rows)} rows are shown; aggregate to see everything."
    return out


def log(entry: dict) -> None:
    with _lock, open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/query":
            return self._send(404, {"error": "not found; POST a Flux query to /query"})
        flux = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8", "replace")
        try:
            max_rows = max(1, min(int(parse_qs(url.query).get("max_rows", ["200"])[0]), MAX_ROWS_CAP))
        except ValueError:
            max_rows = 200
        t0 = time.monotonic()
        problem = check_readonly_flux(flux, BUCKET)
        result = {"error": f"query rejected: {problem}"} if problem else run_query(flux, max_rows)
        log({"ts": time.time(), "flux": flux, "max_rows": max_rows, "rejected": problem,
             "error": result.get("error") if not problem else None,
             "row_count": result.get("row_count", 0), "truncated": result.get("truncated", False),
             "ms": round((time.monotonic() - t0) * 1000)})
        self._send(200, result)

    def log_message(self, *args):                 # quiet: the JSONL log is the record
        pass


if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "a").close()
    ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
