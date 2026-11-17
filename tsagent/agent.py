"""Agent that reads sleep data from the time-series DB and reports on it.

The agent gets a set of tools over the store (daily summaries, a night's hypnogram, data coverage,
re-processing raw EEG, and optionally raw EEG samples) and an LLM (local via Ollama, or Claude)
decides which to call to answer a request.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
from anthropic import beta_tool  # only used to turn functions into tool schemas; works with any backend

from .config import Settings
from .db import Store, check_readonly_flux
from .llm import llm_label, run_loop
from .pipeline import day_window, daterange, get_daily, process_day
from .tracing import TracedTool, Tracer, TracingStore

# Tools given to the agent unless an experiment says otherwise. get_raw_eeg is opt-in.
DEFAULT_TOOLS = ["get_daily_sleep", "get_night_hypnogram", "get_data_coverage", "process_nights"]

SYSTEM_PROMPT = """\
You are a sleep-tracking assistant. The user records EEG overnight; a pipeline stages each 30-second
epoch (W, N1, N2, N3, R) and stores one summary per sleep day in a time-series database. You read
that data through tools and report on it.

Conventions:
- A sleep day is labelled by the date the user woke up and covers 12:00 the previous day to 12:00
  that day, in the user's local time ({tz}).
- "Sleeping hours" means total sleep time (all non-wake epochs), not time in bed.
- Today's date is {today}.

When reporting:
- Lead with the number of hours slept, then the details that matter (efficiency, awakenings,
  deep/REM sleep, comparison with recent nights).
- Only state numbers you got from the tools. If a night has no data, say so; if it has raw EEG but
  no summary, process it first.
- Stage estimates come from an automated classifier, so treat stage minutes as approximate. You are
  not a clinician: don't diagnose; suggest seeing a professional if the user is worried.
"""


QUERY_SYSTEM_PROMPT = """\
You answer the user's questions about their sleep by querying their InfluxDB time-series database
with Flux, using the query_influxdb tool. Decide which queries you need, run them, then answer.

Today's date is {today}. The user's timezone is {tz}. Timestamps in the database are UTC.

Data model: bucket "{bucket}", every point has the tag subject="{subject}".
- sleep_daily: one point per sleep day, timestamped at local midnight of the wake-up date.
  Fields: date (string "YYYY-MM-DD", the wake-up date), total_sleep_h, time_in_bed_h,
  sleep_efficiency (0-1), sleep_onset and final_wake (local time strings "YYYY-MM-DD HH:MM"),
  sleep_latency_min, waso_min (minutes awake after falling asleep), n_awakenings,
  wake_min, n1_min, n2_min, n3_min (deep sleep), rem_min.
- sleep_epoch: one point per 30-second epoch, tag stager. Fields: stage ("W", "N1", "N2", "N3",
  "R"), stage_code (0=W, 1=N1, 2=N2, 3=N3 deep, 4=REM), confidence (0-1).
- eeg_raw: the raw EEG, one point per sample (about 100 per second), tag channel, field value
  (microvolts). Millions of points per night: always aggregate it, never fetch raw samples
  for more than a few seconds.

sleep_epoch and sleep_daily exist only for nights that have already been processed. A night can
have raw EEG in eeg_raw but no summary yet. If you have the process_nights tool, call it for those
nights (it stages the raw EEG and writes sleep_epoch and sleep_daily), then query again. You
cannot stage sleep yourself from raw samples.

Conventions:
- A sleep day is labelled by the wake-up date and runs from 12:00 the previous day to 12:00 that
  day, local time. "Last night" is the sleep day labelled today ({today}).
- "Sleeping hours" means total_sleep_h (time asleep, not time in bed).

Example: sleeping hours for the last 7 sleep days.
from(bucket: "{bucket}")
  |> range(start: -8d)
  |> filter(fn: (r) => r._measurement == "sleep_daily" and r.subject == "{subject}")
  |> filter(fn: (r) => r._field == "date" or r._field == "total_sleep_h")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["date", "total_sleep_h"])

Rules:
- State only numbers that come from query results. If a query fails, read the error, fix the
  query and try again. If there is no data for what was asked, say so plainly.
- Answer in plain language, briefly.
"""


def query_system_prompt(s: Settings, today: date) -> str:
    return QUERY_SYSTEM_PROMPT.format(today=today.isoformat(), tz=s.tz_name,
                                      bucket=s.influx_bucket, subject=s.subject)


def _parse(d: str) -> date:
    return date.fromisoformat(d)


def build_tools(store: Store, s: Settings, names: list[str] | None = None):
    tz = s.tz

    @beta_tool
    def get_daily_sleep(start_date: str, end_date: str) -> str:
        """Get the stored daily sleep summaries for an inclusive date range.

        Each row has: date (wake-up date), total_sleep_h, time_in_bed_h, sleep_efficiency,
        sleep_onset, final_wake, sleep_latency_min, waso_min, n_awakenings and minutes per stage.
        Days with no row have not been processed or have no recording.

        Args:
            start_date: First wake-up date, YYYY-MM-DD.
            end_date: Last wake-up date, YYYY-MM-DD (inclusive).
        """
        df = get_daily(store, _parse(start_date), _parse(end_date), subject=s.subject, tz=tz)
        return df.to_json(orient="records") if not df.empty else "[]"

    @beta_tool
    def get_night_hypnogram(date: str) -> str:
        """Get the sleep-stage timeline for one night, compressed into runs of the same stage.

        Args:
            date: Wake-up date of the night, YYYY-MM-DD.
        """
        start, stop = day_window(_parse(date), tz)
        ep = store.read_epochs(s.subject, start, stop)
        if ep.empty:
            return json.dumps({"date": date, "segments": [], "note": "no staged epochs for this night"})
        out, cur = [], None
        for t, st in zip(ep["time"], ep["stage"]):
            if cur is None or cur["stage"] != st:
                cur = {"stage": st, "start": t.tz_convert(tz).strftime("%H:%M"), "minutes": 0.0}
                out.append(cur)
            cur["minutes"] += 0.5
        return json.dumps({"date": date, "segments": out})

    @beta_tool
    def get_data_coverage(start_date: str, end_date: str) -> str:
        """For each day in an inclusive range, report how many hours of raw EEG are stored and
        whether a daily summary exists. Use this to find nights that need processing.

        Args:
            start_date: First wake-up date, YYYY-MM-DD.
            end_date: Last wake-up date, YYYY-MM-DD (inclusive). At most 62 days after start_date.
        """
        a, b = _parse(start_date), _parse(end_date)
        if (b - a).days > 62:
            return "Error: range too long, use at most 62 days."
        summaries = get_daily(store, a, b, subject=s.subject, tz=tz)
        have = set(summaries["date"]) if not summaries.empty else set()
        rows = []
        for d in daterange(a, b):
            start, stop = day_window(d, tz)
            minutes = store.raw_minutes(s.subject, s.channel, start, stop)
            rows.append({"date": d.isoformat(), "raw_eeg_hours": round(minutes / 60, 2),
                         "has_summary": d.isoformat() in have})
        return json.dumps(rows)

    @beta_tool
    def process_nights(start_date: str, end_date: str) -> str:
        """Run sleep staging on stored raw EEG for each night in an inclusive range and store the
        results, replacing any earlier summary. Returns the new summaries.

        Args:
            start_date: First wake-up date, YYYY-MM-DD.
            end_date: Last wake-up date, YYYY-MM-DD (inclusive). At most 14 days after start_date.
        """
        a, b = _parse(start_date), _parse(end_date)
        if (b - a).days > 14:
            return "Error: range too long, process at most 14 nights per call."
        results = []
        for d in daterange(a, b):
            summary = process_day(store, d, subject=s.subject, channel=s.channel, tz=tz, stager=s.stager)
            results.append(summary or {"date": d.isoformat(), "status": "no raw EEG for this night"})
        return json.dumps(results)

    @beta_tool
    def get_raw_eeg(start_time: str, end_time: str, max_points: int = 2000) -> str:
        """Get raw EEG samples (microvolts) for a time window, evenly downsampled to at most
        max_points values.

        Args:
            start_time: Window start, local time, 'YYYY-MM-DD HH:MM'.
            end_time: Window end, local time, 'YYYY-MM-DD HH:MM'. At most 1 hour after start_time.
            max_points: Maximum number of samples to return, at most 5000.
        """
        import pandas as pd

        a = pd.Timestamp(start_time).tz_localize(tz)
        b = pd.Timestamp(end_time).tz_localize(tz)
        if b - a > pd.Timedelta(hours=1) or b <= a:
            return "Error: the window must be positive and at most 1 hour long."
        segs = store.read_raw(s.subject, s.channel, a, b)
        if not segs:
            return json.dumps({"samples": [], "note": "no raw EEG in this window"})
        values = np.concatenate([seg.values for seg in segs])
        step = max(1, int(np.ceil(len(values) / min(max_points, 5000))))
        return json.dumps({"start": str(segs[0].start.tz_convert(tz)), "unit": "uV",
                           "sampling_rate_hz": segs[0].fs / step,
                           "samples": np.round(values[::step], 1).tolist()})

    @beta_tool
    def query_influxdb(flux: str, max_rows: int = 200) -> str:
        """Run a read-only Flux query on the user's InfluxDB time-series database (sent as an HTTP
        request to the InfluxDB API) and return the result rows as JSON.

        The data model and conventions are described in the system prompt. Only the user's bucket
        may be read; writing with to() and imports other than date, strings, math, regexp,
        timezone and influxdata/influxdb/schema are rejected. Results are cut at max_rows rows,
        so aggregate in Flux (pivot, aggregateWindow, mean, sum, count) instead of fetching raw data.

        Args:
            flux: The complete Flux query.
            max_rows: Maximum number of rows to return, at most 1000.
        """
        problem = check_readonly_flux(flux, s.influx_bucket)
        if problem:
            return f"Error: query rejected: {problem}"
        if not hasattr(store, "flux_query"):
            return "Error: this store does not support Flux queries (InfluxDB only)."
        result = dict(store.flux_query(flux, max_rows=max(1, min(int(max_rows), 1000))))
        result.pop("http", None)  # the HTTP details go to the trace, not to the LLM
        if "error" in result:
            return f"Error: {result['error']}"
        if result["truncated"]:
            result["note"] = f"Only the first {result['row_count']} rows are shown; aggregate to see everything."
        return json.dumps(result, ensure_ascii=False)

    tools = [get_daily_sleep, get_night_hypnogram, get_data_coverage, process_nights, get_raw_eeg,
             query_influxdb]
    names = DEFAULT_TOOLS if names is None else names
    unknown = set(names) - {t.name for t in tools}
    if unknown:
        raise ValueError(f"unknown tool(s): {sorted(unknown)}; available: {[t.name for t in tools]}")
    return [t for t in tools if t.name in names]


def run_agent(prompt: str, store: Store, s: Settings, *, today: date | None = None,
              verbose: bool = False, system: str | None = None, tool_names: list[str] | None = None,
              tracer: Tracer | None = None) -> str:
    """Run the agent on one prompt with the configured LLM backend (local Ollama or Claude).

    Pass a Tracer to record the full information flow; tools and DB access are then traced too.
    """
    today = today or date.today()
    tracer = tracer or Tracer(llm_label(s.llm, s.ollama_url))
    tracer.echo = tracer.echo or verbose
    traced_store = TracingStore(store, tracer)
    tools = [TracedTool(t, tracer) for t in build_tools(traced_store, s, tool_names)]
    if system is None:
        system = SYSTEM_PROMPT.format(tz=s.tz_name, today=today.isoformat())
    return run_loop(backend=s.llm, model=s.llm_model, system=system, prompt=prompt, tools=tools,
                    tracer=tracer, ollama_url=s.ollama_url)


def daily_report_prompt(day: date) -> str:
    week_start = day - timedelta(days=7)
    return (
        f"Write my sleep report for the night ending on {day.isoformat()}. "
        f"If that night has raw EEG but no summary yet, process it first. "
        f"Compare it with the nights from {week_start.isoformat()} to {(day - timedelta(days=1)).isoformat()}. "
        "Keep it short: a headline with the hours slept, then a few bullet points."
    )
