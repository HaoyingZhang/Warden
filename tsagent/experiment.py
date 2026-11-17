"""Run one agent experiment and save its complete information flow.

Input:  a prompt (free text: the task, what data the agent may use, constraints like "preserve my
        privacy"), the tools the agent actually gets, and optionally a system prompt.
Output: experiments/runs/<timestamp>_<name>/
          spec.json    everything needed to reproduce the run
          trace.jsonl  one event per line, full payloads
          answer.md    the agent's final answer
          flow.html    the flow as a sequence diagram + expandable timeline + summary
"""

from __future__ import annotations

import html
import json
import re
from datetime import date, datetime
from pathlib import Path

from .agent import run_agent
from .config import Settings
from .llm import llm_label
from .tracing import Tracer

RUNS_DIR = Path("experiments/runs")

DEFAULT_SYSTEM = (
    "You are an assistant with access to the user's personal data through the tools provided. "
    "Today's date is {today}. The user's timezone is {tz}."
)


def run_experiment(*, prompt: str, name: str, store, s: Settings, tool_names: list[str],
                   system: str | None = None, today: date | None = None, echo: bool = True) -> Path:
    today = today or date.today()
    system = (DEFAULT_SYSTEM if system is None else system).format(today=today.isoformat(), tz=s.tz_name)
    run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{_slug(name)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    spec = {"name": name, "prompt": prompt, "system": system, "tools": tool_names, "today": today.isoformat(),
            "llm_backend": s.llm, "llm_model": s.llm_model, "llm_location": llm_label(s.llm, s.ollama_url),
            "subject": s.subject, "stager": s.stager, "started_at": datetime.now().isoformat(timespec="seconds")}
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False))

    tracer = Tracer(spec["llm_location"])
    tracer.echo = echo
    try:
        answer = run_agent(prompt, store, s, today=today, system=system, tool_names=tool_names, tracer=tracer)
    finally:  # keep the trace even if the run crashes halfway
        (run_dir / "trace.jsonl").write_text(tracer.to_jsonl())
        (run_dir / "flow.html").write_text(render_html(spec, tracer))
    (run_dir / "answer.md").write_text(answer + "\n")
    return run_dir


RESULTS_DIR = Path("experiments/results")


def run_batch(*, tasks_dir: Path, store, s: Settings, tool_names: list[str], system: str | None = None,
              today: date | None = None, out_root: Path = RESULTS_DIR, echo: bool = True) -> Path:
    """Run every task file (*.txt) in tasks_dir and write one JSON result per task.

    Output: <out_root>/<timestamp>/
              <task>.json        question, answer, queries the agent sent, full event trace
              <task>.flow.html   the same flow, for reading in a browser
              summary.json       one line per task
    """
    from dataclasses import asdict

    from .agent import query_system_prompt

    today = today or date.today()
    tasks = sorted(p for p in Path(tasks_dir).glob("*.txt") if p.read_text().strip())
    if not tasks:
        raise SystemExit(f"No non-empty .txt task files in {tasks_dir}")
    if system is None:
        system = query_system_prompt(s, today)
    out_dir = out_root / f"{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for i, task in enumerate(tasks, 1):
        prompt = task.read_text().strip()
        print(f"\n[{i}/{len(tasks)}] {task.name}: {prompt[:80]}", flush=True)
        tracer = Tracer(llm_label(s.llm, s.ollama_url))
        tracer.echo = echo
        status, error, answer = "ok", None, ""
        try:
            answer = run_agent(prompt, store, s, today=today, system=system, tool_names=tool_names,
                               tracer=tracer)
        except Exception as e:  # record the failure and move on to the next task
            status, error = "error", f"{type(e).__name__}: {e}"
        result = {
            "task": task.stem,
            "prompt": prompt,
            "status": status,
            "error": error,
            "answer": answer,
            "today": today.isoformat(),
            "llm": {"backend": s.llm, "model": s.llm_model, "location": tracer.llm_label},
            "tools": tool_names,
            "duration_s": tracer.events[-1].t if tracer.events else 0.0,
            "queries": extract_queries(tracer),
            "flow_summary": summarize_flow(tracer),
            "system_prompt": system,
            "events": [asdict(e) for e in tracer.events],
        }
        (out_dir / f"{task.stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        spec = {"name": task.stem, "prompt": prompt, "system": system, "tools": tool_names,
                "llm_backend": s.llm, "llm_model": s.llm_model, "started_at": datetime.now().isoformat(timespec="seconds")}
        (out_dir / f"{task.stem}.flow.html").write_text(render_html(spec, tracer))
        summary.append({k: result[k] for k in ("task", "prompt", "status", "error", "answer", "duration_s")}
                       | {"n_queries": len(result["queries"])})
        print(f"  → {status}: {(answer or error or '')[:200]}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return out_dir


def extract_queries(tracer: Tracer) -> list[dict]:
    """Pair each query_influxdb call with its outcome: the Flux text, rows returned or the error."""
    out, pending = [], {}
    for e in tracer.events:
        if e.kind == "tool_call" and e.dst == "tool:query_influxdb":
            pending = {"seq": e.seq, "flux": e.payload.get("args", {}).get("flux"),
                       "max_rows": e.payload.get("args", {}).get("max_rows")}
        elif e.kind == "db_read" and e.payload.get("method") == "flux_query" and pending:
            pending["http_request"] = e.payload.get("returned", {}).get("http_request")
        elif e.kind == "tool_result" and e.src == "tool:query_influxdb" and pending:
            content = e.payload.get("content", "")
            if content.startswith("Error"):
                pending.update(ok=False, error=content)
            else:
                try:
                    parsed = json.loads(content)
                    pending.update(ok=True, row_count=parsed.get("row_count"),
                                   truncated=parsed.get("truncated"), rows=parsed.get("rows"))
                except ValueError:
                    pending.update(ok=True, raw=content)
            out.append(pending)
            pending = {}
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", s).strip("-")[:40] or "run"


# --------------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------------

def summarize_flow(tracer: Tracer) -> dict:
    ev = tracer.events
    reqs = [e for e in ev if e.kind == "llm_request"]
    results = [e for e in ev if e.kind == "tool_result"]
    reads = [e for e in ev if e.kind == "db_read"]
    raw_samples = sum(e.payload["returned"].get("samples", 0) for e in reads)
    db_rows = sum(e.payload["returned"].get("rows", 0) for e in reads)
    return {
        "llm_location": tracer.llm_label,
        "leaves_machine": not tracer.llm_label.startswith("local"),
        "llm_requests": len(reqs),
        "bytes_sent_to_llm_total": sum(e.bytes for e in reqs),         # history is resent every turn
        "tool_calls": len(results),
        "tools_used": sorted({e.dst.removeprefix("tool:") for e in ev if e.kind == "tool_call"}),
        "tool_data_to_llm_bytes": sum(e.bytes for e in results),        # personal data the LLM saw
        "db_reads": len(reads),
        "db_raw_samples_read": raw_samples,
        "db_rows_read": db_rows,
        "db_writes": sum(e.kind == "db_write" for e in ev),
    }


# --------------------------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------------------------

def _mermaid(tracer: Tracer) -> str:
    def actor(a: str) -> str:
        return {"user": "User", "agent": "Agent", "llm": "LLM", "db": "DB"}.get(a, a.replace("tool:", "T_"))

    def label(e) -> str:
        text = f"{e.seq}. {e.summary}"
        text = re.sub(r"[^\w\s().,=/→>+-]", " ", text)
        return text[:70] + ("…" if len(text) > 70 else "")

    tools = sorted({a for e in tracer.events for a in (e.src, e.dst) if a.startswith("tool:")})
    where = re.sub(r"[^\w\s().@-]", " ", tracer.llm_label)
    lines = ["sequenceDiagram", "  participant User", "  participant Agent",
             f"  participant LLM as LLM — {where}"]
    lines += [f"  participant {actor(t)} as {t}" for t in tools] + ["  participant DB"]
    for e in tracer.events:
        arrow = "-->>" if e.kind in ("llm_response", "tool_result", "db_read", "answer") else "->>"
        lines.append(f"  {actor(e.src)}{arrow}{actor(e.dst)}: {label(e)}")
    return "\n".join(lines)


KIND_COLORS = {"prompt": "user", "answer": "user", "llm_request": "llm", "llm_response": "llm",
               "tool_call": "tool", "tool_result": "tool", "db_read": "db", "db_write": "db", "error": "err"}


def render_html(spec: dict, tracer: Tracer) -> str:
    summary = summarize_flow(tracer)
    esc = html.escape
    rows = []
    for e in tracer.events:
        body = json.dumps(e.payload, indent=2, ensure_ascii=False)
        if len(body) > 200_000:
            body = body[:200_000] + "\n… (truncated in this view; see trace.jsonl)"
        rows.append(
            f'<details class="ev {KIND_COLORS.get(e.kind, "")}"><summary>'
            f'<span class="seq">{e.seq}</span><span class="t">{e.t:.2f}s</span>'
            f'<span class="kind">{esc(e.kind)}</span>'
            f'<span class="dir">{esc(e.src)} → {esc(e.dst)}</span>'
            f'<span class="sum">{esc(e.summary)}</span><span class="b">{e.bytes:,} B</span>'
            f'</summary><pre>{esc(body)}</pre></details>')
    leave = ("⚠ Everything sent to the LLM left this machine."
             if summary["leaves_machine"] else "✓ The LLM is local: nothing was sent off this machine.")
    stats = "".join(f"<div class='stat'><div class='k'>{esc(k.replace('_', ' '))}</div>"
                    f"<div class='v'>{esc(str(v))}</div></div>" for k, v in summary.items())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Flow — {esc(spec['name'])}</title>
<style>
:root {{ --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --line:#e3e2dd; --card:#ffffff;
  --user:#7c5cff; --llm:#d9480f; --tool:#1c7ed6; --db:#2b8a3e; --err:#c92a2a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --fg:#ecebe6; --muted:#9a9993;
  --line:#2e2e2b; --card:#1f1f1d; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:17px; margin:32px 0 10px; }}
.muted {{ color:var(--muted); }}
pre {{ white-space:pre-wrap; word-break:break-word; font:12.5px/1.45 ui-monospace, monospace;
  background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; margin:6px 0 0; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:8px; }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
.stat .k {{ color:var(--muted); font-size:12px; }} .stat .v {{ font-weight:600; word-break:break-word; }}
.banner {{ padding:10px 14px; border-radius:8px; border:1px solid var(--line); background:var(--card);
  margin:12px 0; font-weight:600; }}
.ev {{ border-left:3px solid var(--line); margin:4px 0; padding:2px 0 2px 10px; }}
.ev.user {{ border-color:var(--user); }} .ev.llm {{ border-color:var(--llm); }}
.ev.tool {{ border-color:var(--tool); }} .ev.db {{ border-color:var(--db); }} .ev.err {{ border-color:var(--err); }}
summary {{ cursor:pointer; display:flex; flex-wrap:wrap; gap:4px 12px; align-items:baseline; }}
.seq {{ color:var(--muted); min-width:2em; text-align:right; font-variant-numeric:tabular-nums; }}
.t, .b {{ color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }}
.kind {{ font:12px ui-monospace, monospace; }} .dir {{ font-weight:600; }}
.sum {{ flex:1 1 300px; }}
.mermaid {{ background:#fff; border-radius:8px; padding:8px; overflow-x:auto; }}
</style></head><body><main>
<h1>{esc(spec['name'])}</h1>
<div class="muted">{esc(spec['started_at'])} · {esc(spec['llm_backend'])} / {esc(spec['llm_model'])}
 · tools: {esc(', '.join(spec['tools']))}</div>
<div class="banner">{esc(leave)}</div>

<h2>Prompt</h2><pre>{esc(spec['prompt'])}</pre>
<h2>System prompt</h2><pre>{esc(spec['system'] or '(none)')}</pre>

<h2>Summary</h2><div class="stats">{stats}</div>

<h2>Sequence</h2>
<div class="mermaid">{esc(_mermaid(tracer))}</div>
<p class="muted">The diagram needs internet access to load its renderer; the timeline below works offline.</p>

<h2>Timeline <span class="muted">(click a row for the full payload)</span></h2>
{''.join(rows)}
</main>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<script>if (window.mermaid) mermaid.initialize({{ startOnLoad: true, theme: "default", sequence: {{ useMaxWidth: false }} }});</script>
</body></html>
"""
