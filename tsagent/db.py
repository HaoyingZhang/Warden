"""Time-series storage.

Three measurements live in one InfluxDB bucket:

  eeg_raw      tags: subject, channel     field: value (microvolts)      one point per sample
  sleep_epoch  tags: subject, stager      fields: stage, stage_code, confidence   one point per 30 s epoch
  sleep_daily  tags: subject              fields: date, total_sleep_h, ...        one point per sleep day

`InfluxStore` talks to InfluxDB 2.x. `MemoryStore` has the same interface and is used in tests
and for dry runs without a database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import numpy as np
import pandas as pd

STAGE_CODES = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
DAILY_FIELDS = [
    "date", "total_sleep_h", "time_in_bed_h", "sleep_efficiency",
    "sleep_onset", "final_wake", "sleep_latency_min", "waso_min",
    "wake_min", "n1_min", "n2_min", "n3_min", "rem_min", "n_awakenings",
]


@dataclass
class RawSignal:
    start: pd.Timestamp  # UTC timestamp of the first sample
    fs: float            # sampling rate (Hz)
    values: np.ndarray   # microvolts


class Store(Protocol):
    def write_raw(self, subject: str, channel: str, start: pd.Timestamp, fs: float, values: np.ndarray) -> None: ...
    def read_raw(self, subject: str, channel: str, start: datetime, stop: datetime) -> list[RawSignal]: ...
    def raw_minutes(self, subject: str, channel: str, start: datetime, stop: datetime) -> int: ...
    def replace_epochs(self, subject: str, start: datetime, stop: datetime, epochs: pd.DataFrame, stager: str) -> None: ...
    def read_epochs(self, subject: str, start: datetime, stop: datetime) -> pd.DataFrame: ...
    def replace_daily(self, subject: str, day_ts: datetime, summary: dict) -> None: ...
    def read_daily(self, subject: str, start: datetime, stop: datetime) -> pd.DataFrame: ...


def split_segments(times_ns: np.ndarray, values: np.ndarray, max_gap_s: float = 2.0) -> list[RawSignal]:
    """Turn sorted (time, value) samples into contiguous segments, splitting at gaps."""
    if len(times_ns) < 2:
        return []
    dt = np.diff(times_ns)
    fs = round(1e9 / float(np.median(dt)), 3)
    breaks = np.where(dt > max_gap_s * 1e9)[0] + 1
    segments = []
    for idx in np.split(np.arange(len(times_ns)), breaks):
        if len(idx) > 1:
            segments.append(RawSignal(pd.Timestamp(int(times_ns[idx[0]]), tz="UTC"), fs, values[idx]))
    return segments


ALLOWED_FLUX_IMPORTS = {"date", "strings", "math", "regexp", "timezone", "influxdata/influxdb/schema"}


def check_readonly_flux(flux: str, bucket: str) -> str | None:
    """Return an error message if a Flux query could write, call out, or read another bucket.

    Flux is a full language: `to()` writes to the DB and packages like `http` or `sql` can send
    data to other servers. Only an allowlist of pure helper packages is accepted. This is a guard,
    not a sandbox: also give the agent a read-only token for real deployments.
    """
    import re

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


def _utc(ts: datetime) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _esc_tag(v: str) -> str:
    return v.replace("\\", "\\\\").replace(",", "\\,").replace("=", "\\=").replace(" ", "\\ ")


def _esc_str(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"')


# --------------------------------------------------------------------------------------------
# InfluxDB
# --------------------------------------------------------------------------------------------

class InfluxStore:
    RAW_CHUNK = 100_000  # lines per write request

    def __init__(self, url: str, token: str, org: str, bucket: str):
        from influxdb_client import InfluxDBClient
        from influxdb_client.client.write_api import SYNCHRONOUS

        self.client = InfluxDBClient(url=url, token=token, org=org, timeout=300_000, enable_gzip=True)
        self.url, self.token = url.rstrip("/"), token
        self.org, self.bucket = org, bucket
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.delete_api = self.client.delete_api()

    @classmethod
    def from_settings(cls, s) -> "InfluxStore":
        return cls(s.influx_url, s.influx_token, s.influx_org, s.influx_bucket)

    def ping(self) -> bool:
        return self.client.ping()

    # ---- helpers ----
    def _write_lines(self, lines: list[str], progress: bool = False) -> None:
        import sys
        import time as _time

        t0 = _time.monotonic()
        for i in range(0, len(lines), self.RAW_CHUNK):
            self.write_api.write(bucket=self.bucket, org=self.org, record=lines[i:i + self.RAW_CHUNK],
                                 write_precision="ns")
            if progress:
                done = min(i + self.RAW_CHUNK, len(lines))
                print(f"\r  written {done:,}/{len(lines):,} samples ({done / len(lines):.0%}, "
                      f"{_time.monotonic() - t0:.0f}s)", end="", file=sys.stderr, flush=True)
        if progress:
            print(file=sys.stderr)

    def _query_csv(self, flux: str) -> pd.DataFrame:
        from io import StringIO

        from influxdb_client import Dialect

        resp = self.query_api.query_raw(flux, org=self.org, dialect=Dialect(header=True, annotations=[]))
        text = resp.data.decode() if hasattr(resp, "data") else resp.read().decode()
        text = text.strip()
        if not text:
            return pd.DataFrame()
        df = pd.read_csv(StringIO(text))
        # Multiple result tables repeat the header; drop those rows.
        if "_time" in df.columns:
            df = df[df["_time"] != "_time"]
        return df

    def _delete(self, start: datetime, stop: datetime, measurement: str, subject: str) -> None:
        self.delete_api.delete(_utc(start), _utc(stop),
                               f'_measurement="{measurement}" AND subject="{subject}"',
                               bucket=self.bucket, org=self.org)

    # ---- free-form Flux over HTTP (used by the agent's query tool) ----
    def flux_query(self, flux: str, max_rows: int = 200) -> dict:
        """POST a Flux query to InfluxDB's HTTP API and return up to max_rows rows.

        The response is streamed and reading stops at max_rows, so a query that would return
        millions of raw samples does not transfer them all. Call check_readonly_flux() first.
        """
        import csv

        import httpx

        request = {
            "method": "POST", "url": f"{self.url}/api/v2/query", "params": {"org": self.org},
            "json": {"query": flux, "type": "flux",
                     "dialect": {"header": True, "annotations": [], "delimiter": ","}},
        }
        headers = {"Authorization": f"Token {self.token}", "Accept": "application/csv"}
        logged = {k: v for k, v in request.items()}  # what the trace shows (token omitted)
        rows: list[dict] = []
        header: list[str] | None = None
        truncated = False
        with httpx.stream(**request, headers=headers, timeout=120) as r:
            if r.status_code >= 400:
                r.read()
                try:
                    msg = r.json().get("message", r.text)
                except ValueError:
                    msg = r.text
                return {"error": f"InfluxDB returned HTTP {r.status_code}: {msg[:1000]}", "http": logged}
            for line in r.iter_lines():
                if not line.strip():          # a blank line separates tables; a new header follows
                    header = None
                    continue
                values = next(csv.reader([line]))
                if header is None:
                    header = values
                    continue
                if len(rows) >= max_rows:
                    truncated = True
                    break
                rows.append({k: v for k, v in zip(header, values) if k not in ("", "result")})
        return {"row_count": len(rows), "truncated": truncated, "rows": rows, "http": logged}

    # ---- raw EEG ----
    def write_raw(self, subject, channel, start, fs, values) -> None:
        t0 = pd.Timestamp(start).tz_convert("UTC").value
        times = t0 + np.round(np.arange(len(values)) * (1e9 / fs)).astype(np.int64)
        prefix = f"eeg_raw,subject={_esc_tag(subject)},channel={_esc_tag(channel)} value="
        lines = [f"{prefix}{v:.3f} {t}" for v, t in zip(values.tolist(), times.tolist())]
        self._write_lines(lines, progress=True)

    def read_raw(self, subject, channel, start, stop) -> list[RawSignal]:
        flux = f'''
from(bucket: "{self.bucket}")
  |> range(start: {_utc(start)}, stop: {_utc(stop)})
  |> filter(fn: (r) => r._measurement == "eeg_raw" and r.subject == "{subject}" and r.channel == "{channel}" and r._field == "value")
  |> keep(columns: ["_time", "_value"])
'''
        # One channel = one series, which InfluxDB streams in time order. Sorting server-side
        # (group + sort) would force it to buffer the whole night in memory, so sort here instead.
        df = self._query_csv(flux)
        if df.empty:
            return []
        times = pd.to_datetime(df["_time"], utc=True, format="ISO8601").astype("int64").to_numpy()
        values = df["_value"].astype(float).to_numpy()
        order = np.argsort(times, kind="stable")
        return split_segments(times[order], values[order])

    def raw_minutes(self, subject, channel, start, stop) -> int:
        """Number of 1-minute windows that contain at least one raw sample."""
        flux = f'''
from(bucket: "{self.bucket}")
  |> range(start: {_utc(start)}, stop: {_utc(stop)})
  |> filter(fn: (r) => r._measurement == "eeg_raw" and r.subject == "{subject}" and r.channel == "{channel}" and r._field == "value")
  |> aggregateWindow(every: 1m, fn: count, createEmpty: false)
  |> count()
  |> group()
  |> sum()
'''
        df = self._query_csv(flux)
        return 0 if df.empty else int(float(df["_value"].iloc[0]))

    # ---- epochs ----
    def replace_epochs(self, subject, start, stop, epochs, stager) -> None:
        self._delete(start, stop, "sleep_epoch", subject)
        tag = f"sleep_epoch,subject={_esc_tag(subject)},stager={_esc_tag(stager)}"
        lines = [
            f'{tag} stage="{r.stage}",stage_code={STAGE_CODES[r.stage]}i,confidence={float(r.confidence):.4f} '
            f'{pd.Timestamp(r.time).value}'
            for r in epochs.itertuples()
        ]
        self._write_lines(lines)

    def read_epochs(self, subject, start, stop) -> pd.DataFrame:
        flux = f'''
from(bucket: "{self.bucket}")
  |> range(start: {_utc(start)}, stop: {_utc(stop)})
  |> filter(fn: (r) => r._measurement == "sleep_epoch" and r.subject == "{subject}")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> group()
  |> keep(columns: ["_time", "stage", "confidence"])
  |> sort(columns: ["_time"])
'''
        df = self._query_csv(flux)
        if df.empty:
            return pd.DataFrame(columns=["time", "stage", "confidence"])
        return pd.DataFrame({
            "time": pd.to_datetime(df["_time"], utc=True, format="ISO8601"),
            "stage": df["stage"].astype(str),
            "confidence": df["confidence"].astype(float),
        }).reset_index(drop=True)

    # ---- daily summaries ----
    def replace_daily(self, subject, day_ts, summary) -> None:
        ts = pd.Timestamp(day_ts)
        self._delete(ts, ts + pd.Timedelta(seconds=1), "sleep_daily", subject)
        fields = []
        for k in DAILY_FIELDS:
            v = summary.get(k)
            if v is None:
                continue
            if isinstance(v, str):
                fields.append(f'{k}="{_esc_str(v)}"')
            elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                fields.append(f"{k}={int(v)}i")
            else:
                fields.append(f"{k}={float(v):.4f}")
        self._write_lines([f"sleep_daily,subject={_esc_tag(subject)} {','.join(fields)} {ts.value}"])

    def read_daily(self, subject, start, stop) -> pd.DataFrame:
        flux = f'''
from(bucket: "{self.bucket}")
  |> range(start: {_utc(start)}, stop: {_utc(stop)})
  |> filter(fn: (r) => r._measurement == "sleep_daily" and r.subject == "{subject}")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> group()
  |> sort(columns: ["_time"])
'''
        df = self._query_csv(flux)
        if df.empty:
            return pd.DataFrame(columns=DAILY_FIELDS)
        cols = [c for c in DAILY_FIELDS if c in df.columns]
        out = df[cols].copy()
        for c in cols:
            if c not in ("date", "sleep_onset", "final_wake"):
                out[c] = pd.to_numeric(out[c])
        return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# In-memory (tests / dry runs)
# --------------------------------------------------------------------------------------------

class MemoryStore:
    def __init__(self):
        self.raw: dict[tuple[str, str], list[RawSignal]] = {}
        self.epochs: dict[str, pd.DataFrame] = {}
        self.daily: dict[str, dict[pd.Timestamp, dict]] = {}

    def write_raw(self, subject, channel, start, fs, values) -> None:
        self.raw.setdefault((subject, channel), []).append(
            RawSignal(pd.Timestamp(start).tz_convert("UTC"), fs, np.asarray(values, float)))

    def _slices(self, subject, channel, start, stop):
        start, stop = pd.Timestamp(start), pd.Timestamp(stop)
        for seg in self.raw.get((subject, channel), []):
            times = seg.start + pd.to_timedelta(np.arange(len(seg.values)) / seg.fs, unit="s")
            mask = (times >= start) & (times < stop)
            if mask.any():
                i0 = int(np.argmax(mask))
                yield RawSignal(times[i0], seg.fs, seg.values[mask])

    def read_raw(self, subject, channel, start, stop):
        return sorted(self._slices(subject, channel, start, stop), key=lambda s: s.start)

    def raw_minutes(self, subject, channel, start, stop) -> int:
        minutes = set()
        for s in self._slices(subject, channel, start, stop):
            times = s.start + pd.to_timedelta(np.arange(len(s.values)) / s.fs, unit="s")
            minutes.update(times.floor("min"))
        return len(minutes)

    def replace_epochs(self, subject, start, stop, epochs, stager) -> None:
        cur = self.epochs.get(subject, pd.DataFrame(columns=["time", "stage", "confidence"]))
        if not cur.empty:
            cur = cur[(cur["time"] < pd.Timestamp(start)) | (cur["time"] >= pd.Timestamp(stop))]
        new = epochs[["time", "stage", "confidence"]]
        self.epochs[subject] = (new if cur.empty else pd.concat([cur, new])).sort_values("time").reset_index(drop=True)

    def read_epochs(self, subject, start, stop) -> pd.DataFrame:
        df = self.epochs.get(subject, pd.DataFrame(columns=["time", "stage", "confidence"]))
        if df.empty:
            return df
        m = (df["time"] >= pd.Timestamp(start)) & (df["time"] < pd.Timestamp(stop))
        return df[m].reset_index(drop=True)

    def replace_daily(self, subject, day_ts, summary) -> None:
        self.daily.setdefault(subject, {})[pd.Timestamp(day_ts)] = dict(summary)

    def read_daily(self, subject, start, stop) -> pd.DataFrame:
        rows = [v for k, v in sorted(self.daily.get(subject, {}).items())
                if pd.Timestamp(start) <= k < pd.Timestamp(stop)]
        return pd.DataFrame(rows, columns=DAILY_FIELDS)
