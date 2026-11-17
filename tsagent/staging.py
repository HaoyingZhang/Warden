"""Sleep staging: raw EEG segment -> one stage per 30 s epoch (W, N1, N2, N3, R).

- `yasa`: YASA's pretrained classifier (Vallat & Walker, 2021). Validated on real PSG; best with a
  central (C4-M1) or frontal (Fpz-Cz) channel over a full night. Use this for real recordings.
- `bandpower`: a transparent rule-based classifier on relative band power. Rough, but needs no
  model and works on short or unusual recordings. Wake-vs-sleep (all that sleep hours depend on)
  is its most reliable output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

from .db import RawSignal

EPOCH_S = 30
MIN_SEGMENT_S = 10 * 60  # shorter segments are not staged
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 15), "beta": (15, 30)}


def _epochs_frame(seg: RawSignal, stages, confidence) -> pd.DataFrame:
    times = seg.start + pd.to_timedelta(np.arange(len(stages)) * EPOCH_S, unit="s")
    return pd.DataFrame({"time": times, "stage": list(stages), "confidence": np.asarray(confidence, float)})


def stage_bandpower(seg: RawSignal) -> pd.DataFrame:
    n = int(EPOCH_S * seg.fs)
    n_epochs = len(seg.values) // n
    x = seg.values[: n_epochs * n].reshape(n_epochs, n)
    x = x - x.mean(axis=1, keepdims=True)
    f, psd = welch(x, fs=seg.fs, nperseg=int(4 * seg.fs), axis=1)
    in_range = (f >= 0.5) & (f <= 30)
    total = psd[:, in_range].sum(axis=1) + 1e-12
    rel = {b: psd[:, (f >= lo) & (f < hi)].sum(axis=1) / total for b, (lo, hi) in BANDS.items()}
    amp = x.std(axis=1)

    stages, conf = [], []
    for i in range(n_epochs):
        d, th, a, s, b = (rel[k][i] for k in ("delta", "theta", "alpha", "sigma", "beta"))
        if d > 0.55 and amp[i] > np.percentile(amp, 60):
            st, c = "N3", d
        elif a > 0.30 or b > 0.20:
            st, c = "W", max(a, b) + 0.3
        elif s > 0.10:
            st, c = "N2", s * 4
        elif b > 0.06:
            st, c = "R", 0.5
        else:
            st, c = "N1", 0.4
        stages.append(st)
        conf.append(min(c, 1.0))

    # Smooth isolated single-epoch flips (A B A -> A A A).
    for i in range(1, n_epochs - 1):
        if stages[i - 1] == stages[i + 1] != stages[i]:
            stages[i] = stages[i - 1]
    return _epochs_frame(seg, stages, conf)


# YASA's labels, in its older (array) and newer (Hypnogram) output, mapped to ours.
_YASA_LABELS = {"W": "W", "WAKE": "W", "N1": "N1", "N2": "N2", "N3": "N3", "R": "R", "REM": "R"}


def stage_yasa(seg: RawSignal, channel: str = "EEG") -> pd.DataFrame:
    import warnings

    import mne
    import yasa

    info = mne.create_info([channel], sfreq=seg.fs, ch_types=["eeg"])
    raw = mne.io.RawArray(seg.values[np.newaxis, :] * 1e-6, info, verbose="ERROR")  # uV -> V
    with warnings.catch_warnings():
        # YASA's pretrained model was saved with an old scikit-learn; loading it in a newer one warns.
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
        sls = yasa.SleepStaging(raw, eeg_name=channel)
        pred = sls.predict()
        if hasattr(pred, "hypno"):  # YASA >= 0.7: a Hypnogram that carries the probabilities
            stages, proba = pred.hypno, pred.proba
        else:  # older YASA: an array of labels
            stages, proba = pred, None
        if proba is None:
            proba = sls.predict_proba()
    labels = [_YASA_LABELS.get(str(s).upper(), str(s)) for s in np.asarray(stages)]
    conf = np.asarray(proba).max(axis=1)
    return _epochs_frame(seg, labels, conf)


def stage_segments(segments: list[RawSignal], method: str = "yasa", channel: str = "EEG") -> pd.DataFrame:
    frames = []
    for seg in segments:
        if len(seg.values) / seg.fs < MIN_SEGMENT_S:
            continue
        if method == "yasa":
            frames.append(stage_yasa(seg, channel))
        elif method == "bandpower":
            frames.append(stage_bandpower(seg))
        else:
            raise ValueError(f"unknown stager {method!r} (use 'yasa' or 'bandpower')")
    if not frames:
        return pd.DataFrame(columns=["time", "stage", "confidence"])
    return pd.concat(frames).sort_values("time").reset_index(drop=True)
