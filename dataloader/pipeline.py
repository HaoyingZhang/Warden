"""Sleep-day definitions: staged 30 s epochs -> one daily summary per sleep day.

A "sleep day" D covers [D-1 12:00, D 12:00) in local time, i.e. it is labelled by the date you
wake up. So the night of Monday->Tuesday is reported as Tuesday. Afternoon naps after 12:00 count
toward the following day. The eval's scenario uses these to build its data and expected answers.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

EPOCH_S = 30
STAGE_CODES = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
DAILY_FIELDS = [
    "date", "total_sleep_h", "time_in_bed_h", "sleep_efficiency",
    "sleep_onset", "final_wake", "sleep_latency_min", "waso_min",
    "wake_min", "n1_min", "n2_min", "n3_min", "rem_min", "n_awakenings",
]


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
