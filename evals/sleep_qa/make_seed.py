"""Write the scenario as InfluxDB line protocol (used when building the sandbox's InfluxDB image).

    python evals/sleep_qa/make_seed.py OUT.lp
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

import scenario as sc  # noqa: E402
from dataloader.pipeline import DAILY_FIELDS, STAGE_CODES  # noqa: E402


def daily_line(n: sc.Night) -> str:
    fields = []
    for k in DAILY_FIELDS:
        v = n.summary.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            fields.append(f'{k}="{v}"')
        elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            fields.append(f"{k}={int(v)}i")
        else:
            fields.append(f"{k}={float(v):.4f}")
    return f"sleep_daily,subject={sc.SUBJECT} {','.join(fields)} {sc.daily_timestamp(n.day).value}"


def main(out: str) -> None:
    n_lines = 0
    with open(out, "w") as f:
        for n in sc.nights():
            tag = f"sleep_epoch,subject={sc.SUBJECT},stager={sc.STAGER}"
            for t, st, c in zip(n.epochs["time"], n.epochs["stage"], n.epochs["confidence"]):
                f.write(f'{tag} stage="{st}",stage_code={STAGE_CODES[st]}i,confidence={c:.4f} {t.value}\n')
                n_lines += 1
            f.write(daily_line(n) + "\n")
            n_lines += 1
        for day in sorted(sc.RAW_NIGHTS):
            start, fs, signal = sc.raw_night(day)
            t0 = start.value
            times = t0 + np.round(np.arange(len(signal)) * (1e9 / fs)).astype(np.int64)
            prefix = f"eeg_raw,subject={sc.SUBJECT},channel=EEG value="
            for v, t in zip(signal.tolist(), times.tolist()):
                f.write(f"{prefix}{v:.3f} {t}\n")
            n_lines += len(signal)
    print(f"wrote {n_lines:,} lines to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "seed.lp")
