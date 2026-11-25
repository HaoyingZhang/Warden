"""Synthetic overnight EEG for development, with a known ground-truth hypnogram.

Each 30 s epoch gets a signal whose spectrum matches its sleep stage:
  W  - alpha (8-12 Hz) and beta activity
  N1 - theta (4-8 Hz), fading alpha
  N2 - theta plus sleep spindles (12-15 Hz bursts) and K-complexes
  N3 - high-amplitude slow waves (0.5-2 Hz)
  R  - low-amplitude mixed theta, sawtooth waves and some beta
on top of 1/f background noise. Good enough to exercise the pipeline; not a substitute for real data.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

EPOCH_S = 30


def synth_hypnogram(rng: np.random.Generator) -> list[str]:
    """A plausible night: sleep cycles with N3 front-loaded and REM growing towards morning."""
    hyp: list[str] = []
    hyp += ["W"] * int(rng.integers(10, 40))      # 5-20 min awake in bed
    hyp += ["N1"] * int(rng.integers(2, 10))
    n_cycles = int(rng.integers(3, 7))
    for c in range(n_cycles):
        frac = c / max(n_cycles - 1, 1)
        hyp += ["N2"] * int(rng.integers(30, 60))
        hyp += ["N3"] * int((1 - frac) * rng.integers(40, 90))
        hyp += ["N2"] * int(rng.integers(20, 40))
        hyp += ["R"] * int(rng.integers(10, 20) + frac * rng.integers(30, 60))
        if rng.random() < 0.5:                        # brief awakening between cycles
            hyp += ["W"] * int(rng.integers(1, 8)) + ["N1"] * int(rng.integers(1, 4))
    hyp += ["W"] * int(rng.integers(10, 40))      # awake in bed in the morning
    return hyp


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / x.std()


def _osc(t, f, amp, rng):
    return amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))


def _bursts(t, fs, f, amp, dur_range, count, rng):
    out = np.zeros_like(t)
    for _ in range(count):
        dur = rng.uniform(*dur_range)
        c = rng.uniform(dur, t[-1] - dur)
        env = np.exp(-0.5 * ((t - c) / (dur / 4)) ** 2)
        out += amp * env * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return out


def synth_epoch(stage: str, fs: float, rng: np.random.Generator) -> np.ndarray:
    n = int(EPOCH_S * fs)
    t = np.arange(n) / fs
    x = 6 * _pink(n, rng)
    if stage == "W":
        x += _osc(t, rng.uniform(9, 11), 18, rng) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.2 * t))
        x += _osc(t, rng.uniform(18, 25), 6, rng) + 4 * rng.standard_normal(n)
    elif stage == "N1":
        x += _osc(t, rng.uniform(5, 7), 14, rng) + _osc(t, 10, 4, rng)
    elif stage == "N2":
        x += _osc(t, rng.uniform(5, 7), 14, rng)
        x += _bursts(t, fs, rng.uniform(12.5, 14), 25, (0.6, 1.5), int(rng.integers(2, 5)), rng)
        x += _bursts(t, fs, 1.0, 60, (0.8, 1.2), int(rng.integers(0, 2)), rng)  # K-complexes
    elif stage == "N3":
        x += _osc(t, rng.uniform(0.7, 1.5), 70, rng) + _osc(t, rng.uniform(1.5, 2.5), 30, rng)
    elif stage == "R":
        x += _osc(t, rng.uniform(5, 7), 10, rng)
        x += _bursts(t, fs, 3.0, 20, (1.0, 3.0), int(rng.integers(1, 4)), rng)
        x += _osc(t, rng.uniform(18, 25), 4, rng)
    else:
        raise ValueError(stage)
    return x


def synth_night(wake_date: date, tz: ZoneInfo, fs: float = 100.0, seed: int | None = None):
    """Generate one night that ends on the morning of `wake_date` (local time).

    Returns (start_utc, fs, signal_uV, hypnogram_df) where hypnogram_df has columns time, stage.
    """
    rng = np.random.default_rng(seed)
    hyp = synth_hypnogram(rng)
    # Lights off between 22:00 and 00:30 the evening before.
    evening = datetime.combine(wake_date - timedelta(days=1), time(22, 0), tzinfo=tz)
    start = pd.Timestamp(evening + timedelta(minutes=float(rng.uniform(0, 150)))).tz_convert("UTC").floor("s")
    signal = np.concatenate([synth_epoch(s, fs, rng) for s in hyp])
    times = start + pd.to_timedelta(np.arange(len(hyp)) * EPOCH_S, unit="s")
    return start, fs, signal, pd.DataFrame({"time": times, "stage": hyp})
