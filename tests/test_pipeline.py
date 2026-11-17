"""Plumbing tests using the in-memory store (no InfluxDB, no API calls)."""

import json
from datetime import date
from zoneinfo import ZoneInfo

import pandas as pd

from tsagent.config import Settings
from tsagent.db import MemoryStore
from tsagent.pipeline import day_window, get_daily, process_day, summarize
from tsagent.synthetic import synth_night

TZ = ZoneInfo("Europe/Paris")


def test_day_window_is_noon_to_noon():
    start, stop = day_window(date(2026, 11, 5), TZ)
    assert (start.day, start.hour, stop.day, stop.hour) == (4, 12, 5, 12)


def test_summarize_counts_sleep_epochs():
    times = pd.date_range("2026-11-04T22:00Z", periods=8, freq="30s")
    ep = pd.DataFrame({"time": times, "stage": ["W", "W", "N1", "N2", "W", "W", "N3", "W"], "confidence": 1.0})
    s = summarize(ep, date(2026, 11, 5), TZ)
    assert s["total_sleep_h"] == round(3 * 30 / 3600, 3)
    assert s["sleep_latency_min"] == 1.0
    assert s["waso_min"] == 1.0
    assert s["n_awakenings"] == 1


def test_process_day_end_to_end():
    store = MemoryStore()
    day = date(2026, 11, 5)
    start, fs, signal, _ = synth_night(day, TZ, seed=1)
    store.write_raw("me", "EEG", start, fs, signal)

    summary = process_day(store, day, subject="me", channel="EEG", tz=TZ, stager="bandpower")
    assert summary is not None and summary["date"] == "2026-11-05"
    assert 0 < summary["total_sleep_h"] <= summary["time_in_bed_h"]

    daily = get_daily(store, day, day, subject="me", tz=TZ)
    assert list(daily["date"]) == ["2026-11-05"]
    assert process_day(store, date(2026, 11, 1), subject="me", channel="EEG", tz=TZ) is None


def test_agent_tools_read_store():
    from tsagent.agent import build_tools

    store = MemoryStore()
    day = date(2026, 11, 5)
    start, fs, signal, _ = synth_night(day, TZ, seed=2)
    store.write_raw("me", "EEG", start, fs, signal)
    s = Settings(subject="me", channel="EEG", tz_name="Europe/Paris", stager="bandpower")
    tools = {t.name: t for t in build_tools(store, s)}

    cov = json.loads(tools["get_data_coverage"].call({"start_date": "2026-11-05", "end_date": "2026-11-05"}))
    assert cov[0]["raw_eeg_hours"] > 0 and cov[0]["has_summary"] is False

    tools["process_nights"].call({"start_date": "2026-11-05", "end_date": "2026-11-05"})
    rows = json.loads(tools["get_daily_sleep"].call({"start_date": "2026-11-05", "end_date": "2026-11-05"}))
    assert rows[0]["total_sleep_h"] > 0
    hyp = json.loads(tools["get_night_hypnogram"].call({"date": "2026-11-05"}))
    assert hyp["segments"]
