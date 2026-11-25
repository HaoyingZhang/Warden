"""Tests for the sleep-day definitions the eval's scenario is built on."""

from datetime import date
from zoneinfo import ZoneInfo

import pandas as pd

from dataloader.pipeline import day_window, summarize

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


def test_summarize_empty_night():
    assert summarize(pd.DataFrame(columns=["time", "stage", "confidence"]), date(2026, 11, 5), TZ) is None
