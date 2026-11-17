"""Raw EEG in the DB -> staged epochs -> one daily summary per sleep day.

A "sleep day" D covers [D-1 12:00, D 12:00) in local time, i.e. it is labelled by the date you
wake up. So the night of Monday->Tuesday is reported as Tuesday. Afternoon naps after 12:00 count
toward the following day.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .db import Store
from .staging import EPOCH_S, stage_segments


def day_window(day: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    end = datetime.combine(day, time(12, 0), tzinfo=tz)
    return end - timedelta(days=1), end


def day_timestamp(day: date, tz: ZoneInfo) -> datetime:
    """Timestamp under which the daily summary is stored: local midnight of the wake date."""
    return datetime.combine(day, time(0, 0), tzinfo=tz)


def summarize(epochs: pd.DataFrame, day: date, tz: ZoneInfo) -> dict | None:
    if epochs.empty:
        return None
    stages = epochs["stage"].to_numpy()
    times = pd.to_datetime(epochs["time"], utc=True)
    asleep = stages != "W"
    minutes = {s: float((stages == s).sum() * EPOCH_S / 60) for s in ("W", "N1", "N2", "N3", "R")}

    summary = {
        "date": day.isoformat(),
        "total_sleep_h": round(asleep.sum() * EPOCH_S / 3600, 3),
        "time_in_bed_h": round(len(stages) * EPOCH_S / 3600, 3),
        "sleep_efficiency": round(float(asleep.mean()), 4),
        "wake_min": minutes["W"], "n1_min": minutes["N1"], "n2_min": minutes["N2"],
        "n3_min": minutes["N3"], "rem_min": minutes["R"],
        "sleep_onset": None, "final_wake": None,
        "sleep_latency_min": None, "waso_min": None, "n_awakenings": 0,
    }
    if asleep.any():
        first = int(np.argmax(asleep))
        last = len(asleep) - 1 - int(np.argmax(asleep[::-1]))
        onset = times.iloc[first]
        final_wake = times.iloc[last] + pd.Timedelta(seconds=EPOCH_S)
        between = stages[first:last + 1]
        # An awakening = a run of >= 2 wake epochs (1 min) after sleep onset.
        runs, run = 0, 0
        for s in between:
            run = run + 1 if s == "W" else 0
            runs += run == 2
        summary.update(
            sleep_onset=onset.tz_convert(tz).strftime("%Y-%m-%d %H:%M"),
            final_wake=final_wake.tz_convert(tz).strftime("%Y-%m-%d %H:%M"),
            sleep_latency_min=round((onset - times.iloc[0]).total_seconds() / 60, 1),
            waso_min=float((between == "W").sum() * EPOCH_S / 60),
            n_awakenings=int(runs),
        )
    return summary


def process_day(store: Store, day: date, *, subject: str, channel: str, tz: ZoneInfo,
                stager: str = "yasa", verbose: bool = False) -> dict | None:
    """Stage the night ending on `day` and store epochs + daily summary. Returns the summary."""
    import sys
    import time as _time

    t0 = _time.monotonic()

    def log(msg: str) -> None:
        if verbose:
            print(f"  [{_time.monotonic() - t0:5.1f}s] {msg}", file=sys.stderr, flush=True)

    start, stop = day_window(day, tz)
    log(f"reading raw EEG {start:%Y-%m-%d %H:%M} -> {stop:%Y-%m-%d %H:%M} ...")
    segments = store.read_raw(subject, channel, start, stop)
    if not segments:
        log("no raw EEG in this window")
        return None
    n = sum(len(s.values) for s in segments)
    log(f"read {n:,} samples in {len(segments)} segment(s); staging with {stager} ...")
    epochs = stage_segments(segments, method=stager, channel=channel)
    epochs = epochs[(epochs["time"] >= pd.Timestamp(start)) & (epochs["time"] < pd.Timestamp(stop))]
    if epochs.empty:
        log("no epochs staged (recording shorter than 10 min?)")
        return None
    log(f"staged {len(epochs)} epochs; writing results ...")
    store.replace_epochs(subject, start, stop, epochs, stager)
    summary = summarize(epochs, day, tz)
    store.replace_daily(subject, day_timestamp(day, tz), summary)
    log("done")
    return summary


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def get_daily(store: Store, start: date, end: date, *, subject: str, tz: ZoneInfo) -> pd.DataFrame:
    return store.read_daily(subject, day_timestamp(start, tz), day_timestamp(end + timedelta(days=1), tz))
