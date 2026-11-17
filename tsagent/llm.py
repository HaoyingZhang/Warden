"""The agent loop: ask the LLM, run the tools it requests, send results back, repeat.

Two backends with the same loop and the same tracing:
  ollama     - a local model served by Ollama (default; free, nothing leaves the machine)
  anthropic  - Claude via the Anthropic API (needs a paid API key)
"""

from __future__ import annotations

import json
from urllib.parse import urlparse

from .tracing import Tracer, jsonable, nbytes
import httpx
import anthropic


def llm_label(backend: str, ollama_url: str) -> str:
    if backend == "anthropic":
        return "cloud (api.anthropic.com)"
    host = urlparse(ollama_url).hostname or ollama_url
    # "ollama" is the Docker Compose service on the same machine; host.docker.internal is the host itself.
    local_hosts = ("localhost", "127.0.0.1", "::1", "ollama", "host.docker.internal")
    where = "local" if host in local_hosts else "remote"
    return f"{where} (ollama @ {host})"


def run_loop(*, backend: str, model: str, system: str, prompt: str, tools: list, tracer: Tracer,
             ollama_url: str = "http://localhost:11434", max_iterations: int = 20) -> str:
    tracer.emit("prompt", "user", "agent", f"prompt ({len(prompt):,} chars)", {"prompt": prompt, "system": system})
    if backend == "ollama":
        answer = _run_ollama(model, system, prompt, tools, tracer, ollama_url, max_iterations)
    elif backend == "anthropic":
        answer = _run_anthropic(model, system, prompt, tools, tracer, max_iterations)
    else:
        raise ValueError(f"unknown LLM backend {backend!r} (use 'ollama' or 'anthropic')")
    tracer.emit("answer", "agent", "user", f"answer ({len(answer):,} chars)", {"text": answer})
    return answer


def _log_request(tracer: Tracer, request: dict, sent_before: int) -> None:
    """Log one LLM request: full payload, plus which messages are new since the last request."""
    msgs = request["messages"]
    tracer.emit(
        "llm_request", "agent", "llm",
        f"request #{sum(e.kind == 'llm_request' for e in tracer.events) + 1}: "
        f"{len(msgs)} messages ({len(msgs) - sent_before} new)",
        {"new_messages": msgs[sent_before:], "full_request": request},
        size=nbytes(request),
    )


# --------------------------------------------------------------------------------------------
# Ollama (local)
# --------------------------------------------------------------------------------------------

def _run_ollama(model, system, prompt, tools, tracer, url, max_iterations) -> str:
    by_name = {t.name: t for t in tools}
    tool_defs = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                   "parameters": t.input_schema}} for t in tools]
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    sent = 0
    for _ in range(max_iterations):
        request = {"model": model, "messages": messages, "tools": tool_defs, "stream": False}
        _log_request(tracer, request, sent)
        sent = len(messages)
        try:
            r = httpx.post(f"{url.rstrip('/')}/api/chat", json=request, timeout=900)
            r.raise_for_status()
        except httpx.HTTPError as e:
            tracer.emit("error", "llm", "agent", f"Ollama request failed: {e}", {"error": str(e)})
            raise SystemExit(f"Ollama request failed ({e}). Is Ollama running and is the model pulled? "
                             f"(Docker: `docker compose exec ollama ollama pull {model}`)")
        msg = r.json()["message"]
        calls = msg.get("tool_calls") or []
        tracer.emit("llm_response", "llm", "agent",
                    f"{len(calls)} tool call(s)" if calls else f"text ({len(msg.get('content', '')):,} chars)",
                    {"content": msg.get("content", ""), "thinking": msg.get("thinking"), "tool_calls": calls})
        messages.append(msg)
        if not calls:
            return (msg.get("content") or "").strip()
        for call in calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            out = by_name[name].call(args) if name in by_name else f"Error: unknown tool {name!r}"
            messages.append({"role": "tool", "tool_name": name, "content": out})
    return "(stopped: too many iterations)"


# --------------------------------------------------------------------------------------------
# Anthropic (cloud)
# --------------------------------------------------------------------------------------------

def _run_anthropic(model, system, prompt, tools, tracer, max_iterations) -> str:
    client = anthropic.Anthropic()
    by_name = {t.name: t for t in tools}
    messages: list = [{"role": "user", "content": prompt}]
    sent = 0
    for _ in range(max_iterations):
        request = dict(
            model=model, max_tokens=16000, messages=messages, **({"system": system} if system else {}),
            tools=[t.to_dict() for t in tools],
            thinking={"type": "adaptive"}, output_config={"effort": "medium"},
            # If a safety classifier declines, the API retries on a fallback model automatically.
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        _log_request(tracer, jsonable(request), sent)
        sent = len(messages)
        resp = client.beta.messages.create(**request)
        uses = [b for b in resp.content if b.type == "tool_use"]
        text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
        tracer.emit("llm_response", "llm", "agent",
                    f"{len(uses)} tool call(s)" if uses else f"text ({len(text):,} chars)",
                    {"content": resp.content, "stop_reason": resp.stop_reason, "usage": resp.usage})
        if resp.stop_reason == "refusal":
            return "The model declined this request."
        messages.append({"role": "assistant", "content": resp.content})
        if not uses:
            return text
        results = []
        for b in uses:
            out = by_name[b.name].call(b.input) if b.name in by_name else f"Error: unknown tool {b.name!r}"
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out,
                            "is_error": out.startswith("Error:")})
        messages.append({"role": "user", "content": results})  # all results in one message
    return "(stopped: too many iterations)"
