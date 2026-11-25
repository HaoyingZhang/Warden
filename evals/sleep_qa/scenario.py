"""The fixed, synthetic sleep history the eval runs against.

This one module is used twice, so the data and the expected answers can never drift apart:
  - make_seed.py turns it into InfluxDB line protocol, baked into the sandbox's InfluxDB image;
  - build_dataset.py computes the ground truth for each question from it.

Everything is deterministic (fixed seed). Change SEED, TODAY or MISSING, then rebuild the dataset
(python evals/sleep_qa/build_dataset.py) and the image is rebuilt on the next `inspect eval`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from dataloader.pipeline import day_timestamp, day_window, summarize
from dataloader.synthetic import EPOCH_S, synth_hypnogram, synth_night

SEED = 20261117
TZ_NAME = "Europe/Paris"
TZ = ZoneInfo(TZ_NAME)
SUBJECT = "me"
BUCKET = "eeg"
STAGER = "yasa"

TODAY = date(2026, 11, 17)                        # the user's "today" in every question
NOW_UTC = "2026-11-17T10:00:00Z"                  # Flux now() inside the sandbox (11:00 in Paris)
N_NIGHTS = 14                                     # wake dates TODAY-13 .. TODAY
MISSING = {date(2026, 11, 14)}                    # no recording that night (sensor not worn)
RAW_NIGHTS = {TODAY}                              # nights that also have raw EEG in eeg_raw
RAW_FS = 10.0                                     # Hz; low on purpose to keep the image small


@dataclass
class Night:
    day: date                                     # wake-up date (the sleep day's label)
    epochs: pd.DataFrame                          # time (UTC), stage, confidence
    summary: dict                                 # same fields as sleep_daily


def wake_dates() -> list[date]:
    return [TODAY - timedelta(days=i) for i in range(N_NIGHTS - 1, -1, -1)]


def _night(day: date, seed: int) -> Night:
    # Same random sequence as dataloader.synthetic.synth_night: hypnogram first, then the start time.
    rng = np.random.default_rng(seed)
    hyp = synth_hypnogram(rng)
    evening = datetime.combine(day - timedelta(days=1), time(22, 0), tzinfo=TZ)
    start = pd.Timestamp(evening + timedelta(minutes=float(rng.uniform(0, 150)))).tz_convert("UTC").floor("s")
    times = start + pd.to_timedelta(np.arange(len(hyp)) * EPOCH_S, unit="s")
    conf = np.round(np.random.default_rng(seed + 1_000_000).uniform(0.55, 0.99, len(hyp)), 4)
    ep = pd.DataFrame({"time": times, "stage": hyp, "confidence": conf})
    lo, hi = day_window(day, TZ)                  # as the pipeline does: keep the noon-to-noon window
    ep = ep[(ep["time"] >= pd.Timestamp(lo)) & (ep["time"] < pd.Timestamp(hi))].reset_index(drop=True)
    return Night(day, ep, summarize(ep, day, TZ))


def nights() -> list[Night]:
    """Every recorded night, oldest first (missing nights are left out)."""
    return [_night(d, SEED + i) for i, d in enumerate(wake_dates()) if d not in MISSING]


def raw_night(day: date) -> tuple[pd.Timestamp, float, np.ndarray]:
    i = wake_dates().index(day)
    start, fs, signal, _ = synth_night(day, TZ, fs=RAW_FS, seed=SEED + i)
    return start, fs, signal


def daily_timestamp(day: date) -> pd.Timestamp:
    return pd.Timestamp(day_timestamp(day, TZ))
