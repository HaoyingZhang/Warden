"""Records the full information flow of one agent run.

Actors:  user · agent (the loop on this machine) · llm · tool:<name> · db
Every message between actors becomes an Event with the complete payload, so a run can be audited
afterwards: what the LLM saw, which data each tool read from the DB, and what it passed on.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def jsonable(obj: Any) -> Any:
    """Convert SDK models, numpy and pandas values into plain JSON-compatible data."""
    if hasattr(obj, "model_dump"):
        return jsonable(obj.model_dump(mode="json", exclude_none=True))
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def nbytes(obj: Any) -> int:
    return len(json.dumps(jsonable(obj), ensure_ascii=False).encode())


@dataclass
class Event:
    seq: int
    t: float          # seconds since the run started
    kind: str         # prompt | llm_request | llm_response | tool_call | db_read | db_write | tool_result | answer | error
    src: str
    dst: str
    summary: str
    payload: dict = field(default_factory=dict)
    bytes: int = 0


class Tracer:
    def __init__(self, llm_label: str):
        self.events: list[Event] = []
        self.t0 = time.monotonic()
        self.llm_label = llm_label      # e.g. "local (ollama @ localhost:11434)" or "cloud (api.anthropic.com)"
        self.current_tool: str | None = None
        self.echo = False               # print events as they happen

    def emit(self, kind: str, src: str, dst: str, summary: str, payload: dict | None = None,
             size: int | None = None) -> Event:
        payload = jsonable(payload or {})
        ev = Event(len(self.events) + 1, round(time.monotonic() - self.t0, 3), kind, src, dst, summary,
                   payload, nbytes(payload) if size is None else size)
        self.events.append(ev)
        if self.echo:
            print(f"  {ev.seq:3d} {ev.t:7.2f}s  {src:>18} → {dst:<18} {summary}  ({ev.bytes:,} B)", flush=True)
        return ev

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(asdict(e), ensure_ascii=False) for e in self.events) + "\n"


# --------------------------------------------------------------------------------------------
# Store wrapper: logs every DB access, attributed to the tool that made it
# --------------------------------------------------------------------------------------------

def _size_of(result: Any) -> dict:
    if isinstance(result, dict) and ("rows" in result or "error" in result):  # flux_query
        info = {"http_request": result.get("http")}
        if "error" in result:
            info["error"] = result["error"]
        else:
            info.update(rows=result["row_count"], truncated=result["truncated"])
        return info
    if isinstance(result, pd.DataFrame):
        return {"rows": len(result), "columns": list(result.columns)}
    if isinstance(result, list) and result and hasattr(result[0], "values"):  # list[RawSignal]
        return {"segments": len(result), "samples": int(sum(len(s.values) for s in result)),
                "fs_hz": result[0].fs}
    if isinstance(result, list):
        return {"items": len(result)}
    return {"value": result}


class TracingStore:
    READS = {"read_raw", "read_epochs", "read_daily", "raw_minutes", "flux_query"}
    WRITES = {"write_raw", "replace_epochs", "replace_daily"}

    def __init__(self, inner, tracer: Tracer):
        self._inner, self._tracer = inner, tracer

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name not in self.READS | self.WRITES or not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            result = attr(*args, **kwargs)
            actor = f"tool:{self._tracer.current_tool}" if self._tracer.current_tool else "agent"
            shown_args = [a for a in args if not isinstance(a, (pd.DataFrame, np.ndarray, dict))]
            if name in self.READS:
                info = _size_of(result)
                self._tracer.emit("db_read", "db", actor, f"{name} → {_short(info)}",
                                  {"method": name, "args": shown_args, "kwargs": kwargs, "returned": info},
                                  size=0)
            else:
                if name == "replace_epochs":
                    written = {"epochs": len(args[3])}
                elif name == "replace_daily":
                    written = args[2]  # the daily summary dict
                else:
                    written = {"samples": len(args[4])}
                self._tracer.emit("db_write", actor, "db", name,
                                  {"method": name, "args": shown_args, "written": written}, size=0)
            return result

        return wrapped


def _short(info: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in info.items() if k not in ("columns", "http_request"))


# --------------------------------------------------------------------------------------------
# Tool wrapper: logs the call and the result handed back to the agent (and then to the LLM)
# --------------------------------------------------------------------------------------------

class TracedTool:
    def __init__(self, tool, tracer: Tracer):
        self._tool, self._tracer = tool, tracer
        self.name = tool.name
        self.description = tool.description
        self.input_schema = tool.input_schema

    def to_dict(self):
        return self._tool.to_dict()

    def call(self, args: dict) -> str:
        tr = self._tracer
        tr.emit("tool_call", "agent", f"tool:{self.name}", f"{self.name}({_args(args)})", {"args": args})
        tr.current_tool = self.name
        try:
            out = self._tool.call(args)
            out = out if isinstance(out, str) else json.dumps(jsonable(out))
            is_error = False
        except Exception as e:  # report tool failures to the model instead of crashing the run
            out, is_error = f"Error: {type(e).__name__}: {e}", True
        finally:
            tr.current_tool = None
        tr.emit("tool_result", f"tool:{self.name}", "agent",
                f"{self.name} returned {len(out):,} chars" + (" (error)" if is_error else ""),
                {"content": out, "is_error": is_error}, size=len(out.encode()))
        return out


def _args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())
