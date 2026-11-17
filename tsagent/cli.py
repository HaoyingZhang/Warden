"""Command-line entry point: python -m tsagent <command> ..."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .config import settings as S
from .db import InfluxStore
from .pipeline import daterange, get_daily, process_day
from .synthetic import synth_night
import mne
from .agent import daily_report_prompt, run_agent
from .agent import run_agent
from .agent import DEFAULT_TOOLS
from .experiment import run_batch, run_experiment


REPORTS_DIR = Path("reports")


def _date(v: str) -> date:
    return date.fromisoformat(v)


def _store() -> InfluxStore:
    store = InfluxStore.from_settings(S)
    if not store.ping():
        sys.exit(f"Cannot reach InfluxDB at {S.influx_url}. Is it running? (docker compose up -d)")
    return store


def cmd_simulate(a):
    store = _store()
    end = a.end or date.today()
    for i, d in enumerate(daterange(end - timedelta(days=a.days - 1), end)):
        print(f"{d}: generating night...", flush=True)
        start, fs, signal, hyp = synth_night(d, S.tz, fs=a.fs, seed=None if a.seed is None else a.seed + i)
        store.write_raw(S.subject, S.channel, start, fs, signal)
        truth_h = (hyp["stage"] != "W").sum() * 30 / 3600 # each stage is for 30 s
        print(f"{d}: wrote {len(signal) / fs / 3600:.1f} h of EEG at {fs:g} Hz (true sleep {truth_h:.2f} h)")


def cmd_process(a):
    store = _store()
    start = a.start or (date.today() if a.end is None else a.end)
    for d in daterange(start, a.end or start):
        print(f"{d}: processing...", flush=True)
        s = process_day(store, d, subject=S.subject, channel=S.channel, tz=S.tz,
                        stager=a.stager or S.stager, verbose=True)
        print(f"{d}: " + (f"{s['total_sleep_h']:.2f} h asleep ({s['sleep_onset']} -> {s['final_wake']})"
                          if s else "no raw EEG"))


def cmd_hours(a):
    store = _store()
    end = a.end or date.today()
    df = get_daily(store, a.start or end - timedelta(days=6), end, subject=S.subject, tz=S.tz)
    if df.empty:
        print("No daily summaries in that range. Run `process` first.")
        return
    print(df[["date", "total_sleep_h", "time_in_bed_h", "sleep_efficiency", "sleep_onset", "final_wake"]]
          .to_string(index=False))


def cmd_report(a):
    day = a.date or date.today()
    text = run_agent(daily_report_prompt(day), _store(), S, verbose=a.verbose)
    print(text)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f"{day.isoformat()}.md").write_text(text + "\n")


def cmd_ask(a):
    print(run_agent(" ".join(a.question), _store(), S, verbose=a.verbose))


def cmd_experiment(a):
    prompt_path = Path(a.prompt_file)
    prompt = prompt_path.read_text().strip()
    system = Path(a.system_file).read_text().strip() if a.system_file else None
    tools = [t.strip() for t in a.tools.split(",") if t.strip()] if a.tools else DEFAULT_TOOLS
    print(f"Running '{prompt_path.stem}' with {S.llm}/{S.llm_model}, tools: {', '.join(tools)}\n")
    run_dir = run_experiment(prompt=prompt, name=a.name or prompt_path.stem, store=_store(), s=S,
                             tool_names=tools, system=system, today=a.today)
    print(f"\nAnswer:\n{(run_dir / 'answer.md').read_text()}")
    print(f"Saved to {run_dir}/  (open flow.html in a browser)")


def cmd_batch(a):
    tools = [t.strip() for t in a.tools.split(",") if t.strip()]
    system = Path(a.system_file).read_text().strip() if a.system_file else None
    print(f"Running tasks in {a.tasks_dir} with {S.llm}/{S.llm_model}, tools: {', '.join(tools)}")
    out_dir = run_batch(tasks_dir=Path(a.tasks_dir), store=_store(), s=S, tool_names=tools,
                        system=system, today=a.today, out_root=Path(a.out))
    print(f"\nResults written to {out_dir}/  (summary.json, one <task>.json per task)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="tsagent", description="EEG sleep-hours agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("simulate", help="write synthetic overnight EEG into the DB")
    sp.add_argument("--days", type=int, default=7)
    sp.add_argument("--end", type=_date, help="last wake-up date (default: today)")
    sp.add_argument("--fs", type=float, default=100.0)
    sp.add_argument("--seed", type=int)
    sp.set_defaults(fn=cmd_simulate)

    sp = sub.add_parser("process", help="stage raw EEG and store daily summaries")
    sp.add_argument("--start", type=_date)
    sp.add_argument("--end", type=_date)
    sp.add_argument("--stager", choices=["yasa", "bandpower"])
    sp.set_defaults(fn=cmd_process)

    sp = sub.add_parser("hours", help="print daily sleep hours (no LLM)")
    sp.add_argument("--start", type=_date)
    sp.add_argument("--end", type=_date)
    sp.set_defaults(fn=cmd_hours)

    sp = sub.add_parser("report", help="Claude writes the daily sleep report")
    sp.add_argument("--date", type=_date, help="wake-up date (default: today)")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(fn=cmd_report)

    sp = sub.add_parser("ask", help="ask the agent a question about your sleep")
    sp.add_argument("question", nargs="+")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(fn=cmd_ask)

    sp = sub.add_parser("experiment", help="run a prompt and record the full information flow")
    sp.add_argument("prompt_file", help="text file with the prompt")
    sp.add_argument("--tools", help="comma-separated tools the agent gets (default: all except get_raw_eeg). "
                                    "Available: get_daily_sleep, get_night_hypnogram, get_data_coverage, "
                                    "process_nights, get_raw_eeg, query_influxdb")
    sp.add_argument("--system-file", help="custom system prompt file (an empty file = no system prompt)")
    sp.add_argument("--name", help="run name (default: prompt file name)")
    sp.add_argument("--today", type=_date, help="date the agent believes is today (for reproducibility)")
    sp.set_defaults(fn=cmd_experiment)

    sp = sub.add_parser("batch", help="run every task file in a folder; the agent writes its own DB queries")
    sp.add_argument("tasks_dir", nargs="?", default="experiments/tasks", help="folder of .txt task files")
    sp.add_argument("--tools", default="query_influxdb,process_nights",
                    help="comma-separated tools (default: query_influxdb,process_nights)")
    sp.add_argument("--system-file", help="custom system prompt file (default: the DB schema prompt)")
    sp.add_argument("--today", type=_date, help="date the agent believes is today (for reproducibility)")
    sp.add_argument("--out", default="experiments/results", help="folder for the JSON results")
    sp.set_defaults(fn=cmd_batch)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
