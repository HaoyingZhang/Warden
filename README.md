# TSAgent: EEG → daily sleeping hours

Overnight EEG goes into InfluxDB. A pipeline splits it into 30 s epochs and labels each one with a
sleep stage (W, N1, N2, N3, R). It then stores one summary per day, including total sleep hours. An
LLM agent (a local model, or Claude) reads those summaries through tools and writes the daily report
or answers questions. Experiments record the agent's full information flow.

```
EEG (synthetic / EDF) ──► InfluxDB: eeg_raw ──► staging ──► sleep_epoch ──► sleep_daily ──► LLM agent ──► report
```

## Quick start: run an experiment with Docker

Run these from the project folder (`cd /home/haoying/Documents/TSAgent`), with Docker Desktop running.

**0. One-time setup**
```bash
docker compose up -d                                  # start InfluxDB and Ollama (the local LLM)
docker compose exec ollama ollama pull qwen3:8b       # download the model (~5 GB)
docker compose build agent                            # build the agent image (again after changing requirements.txt)
```

**1. Start InfluxDB and Ollama** (if they aren't already running)
```bash
docker compose up -d
```

**2. Write synthetic raw EEG** (only raw data; nothing is processed yet)
```bash
docker compose run --rm agent simulate --days 3 --end 2026-11-17                  # raw EEG for 3 nights
```

**3. Run the agent on your requests** (it starts, answers, saves the results and stops)

The agent finds out by itself that the nights are not processed yet, and calls `process_nights`
(sleep staging) before answering. The LLM decides when to call the staging tool, but the staging is
done by the classifier in `tsagent/staging.py`, not by the LLM.
```bash
# every question in experiments/tasks/*.txt → experiments/results/<timestamp>/*.json
docker compose run --rm agent batch --today 2026-11-17

# or one prompt, with the full flow report → experiments/runs/<timestamp>_<name>/flow.html
docker compose run --rm agent experiment experiments/prompts/sleep_privacy.txt --today 2026-11-17
```

**4. Look at the results**
- `experiments/results/<timestamp>/summary.json`: question, answer and number of queries for each task
- `experiments/results/<timestamp>/<task>.json`: the queries the agent sent and the full trace
- `experiments/results/<timestamp>/<task>.flow.html`: open in a browser
- http://localhost:8086: the data in InfluxDB

**When you're done**
```bash
docker compose down              # stop; data and the model are kept
```

To add a question, create a new `.txt` file in `experiments/tasks/`. Code changes in `tsagent/`
apply immediately; no rebuild is needed. Steps 0 and 2 are only needed once, unless you want new data.

## Setup (without Docker, for development)
Set the username and password for Influxdb client by editing the docker-compose.yml:
- DOCKER_INFLUXDB_INIT_USERNAME: ${INFLUX_ADMIN_USER:-[MY_USER_NAME]}
- DOCKER_INFLUXDB_INIT_PASSWORD: ${INFLUX_ADMIN_PASSWORD:-[MY_PASSWORD]}

```bash
docker compose up -d              # InfluxDB 2.7 on http://localhost:8086
python3 -m venv .venv
source .venv/bin/activate
(.venv) pip install -r requirements.txt
```

## Usage

```bash
(.venv)python -m tsagent simulate --days 3          # synthetic EEG for the last 3 nights
(.venv)python -m tsagent process --start 2026-11-06 --end 2026-11-06 --stager bandpower
.venv/bin/python -m tsagent hours                       # table of daily sleep hours (no LLM)
.venv/bin/python -m tsagent report -v                   # Claude writes today's report → reports/YYYY-MM-DD.md
(.venv)python -m tsagent ask "Which night this week did I sleep the least, and why?"
.venv/bin/python -m tsagent load-edf night.edf --edf-channel "EEG Fpz-Cz"   # real recordings later
.venv/bin/python -m pytest tests                        # plumbing tests, no DB or API needed
```

Daily automation (cron, 09:00 every day):

```
0 9 * * * cd /home/haoying/Documents/TSAgent && .venv/bin/python -m tsagent report >> reports/cron.log 2>&1
```

The report command processes the night itself if it hasn't been processed yet, so `process` doesn't
need its own cron job.

## Definitions

- **Sleep day D** = 12:00 on D-1 to 12:00 on D, local time (`TSAGENT_TZ`). Each night is labelled
  with the date you wake up.
- **Sleeping hours** = number of non-wake epochs × 30 s (total sleep time, not time in bed).

## Data model (InfluxDB bucket `eeg`)

| measurement   | tags               | fields                                                             | resolution |
|---------------|--------------------|--------------------------------------------------------------------|------------|
| `eeg_raw`     | subject, channel   | value (µV)                                                         | per sample |
| `sleep_epoch` | subject, stager    | stage, stage_code, confidence                                      | 30 s       |
| `sleep_daily` | subject            | date, total_sleep_h, time_in_bed_h, sleep_efficiency, onset/wake, latency, WASO, awakenings, minutes per stage | 1 per day |

## Where to plug in your own logic

| file | what it does |
|------|--------------|
| `tsagent/staging.py` | `stage_segments()`: raw EEG → stage per epoch. Placeholder stagers: `yasa` (pretrained) and `bandpower` (rules). Replace with your own. |
| `tsagent/pipeline.py` | `summarize()`: epochs → daily metrics. Change how "sleeping hours" is defined here. |
| `tsagent/synthetic.py` | Fake EEG generator for development. |
| `tsagent/db.py` | `InfluxStore` and an in-memory `MemoryStore` with the same interface. |
| `tsagent/agent.py` | The agent's tools and system prompt: `get_daily_sleep`, `get_night_hypnogram`, `get_data_coverage`, `process_nights`, and the opt-in `get_raw_eeg`. |
| `tsagent/llm.py` | The agent loop, for a local model (Ollama) or Claude (Anthropic API). |
| `tsagent/tracing.py` | Records every message between user, agent, LLM, tools and DB. |
| `tsagent/experiment.py` | Runs one experiment and writes the trace and the flow report. |

With the Claude backend, the agent enables server-side refusal fallbacks (`fallbacks="default"`):
if a safety classifier declines a request, the API retries it on another model automatically.
Remove the `betas` and `fallbacks` arguments in `llm.py` to turn that off.

## Running the agent in Docker (isolated)

All three parts run as containers. The agent sits on an internal Docker network with **no internet
access**: it can reach only InfluxDB and the local LLM.

```
 backend network (internal, no internet)          default network (internet)
 ┌──────────┐   ┌──────────┐   ┌──────────┐
 │  agent   │──►│ influxdb │   │  ollama  │◄── model downloads
 │          │──────────────────►│          │
 └──────────┘   └────┬─────┘   └──────────┘
                     └── web UI on 127.0.0.1:8086
```

```bash
docker compose up -d                                   # InfluxDB + Ollama
docker compose exec ollama ollama pull qwen3:8b        # one-time model download (~5 GB)
docker compose build agent                             # build the agent image (rebuild after changing requirements.txt)

docker compose run --rm agent simulate --days 3
docker compose run --rm agent process --start 2026-11-14 --end 2026-11-16
docker compose run --rm agent experiment experiments/prompts/sleep_privacy.txt
```

The agent container:
- has no internet and sees only the variables it needs (not the InfluxDB admin password, not API keys);
- can read only the code (read-only) and `experiments/prompts/`, and write only to `experiments/runs/`
  and `reports/`. Put prompt files in `experiments/prompts/`;
- runs as a non-root user, with a read-only filesystem and no Linux capabilities.

To compare with Claude, which needs internet, add the override file. That removes the isolation:
```bash
docker compose -f docker-compose.yml -f docker-compose.cloud.yml run --rm agent experiment experiments/prompts/sleep_privacy.txt
```

## LLM backend

Default: a local model through [Ollama](https://ollama.com). Nothing leaves your machine and there
is no API cost. In Docker it runs as the `ollama` service (above). Without Docker:

```bash
curl -fsSL https://ollama.com/install.sh | sh     # install Ollama (runs as a service)
ollama pull qwen3:8b                              # ~5 GB; needs a model that supports tool calling
```

Set `TSAGENT_LLM=anthropic` in `.env` to use Claude instead (needs a paid API key).

## Information-flow experiments

Write a prompt in a text file: the task, which data the agent may use, any constraints ("preserve my
privacy"). Then run it and choose which tools the agent actually gets:

```bash
python -m tsagent experiment experiments/prompts/sleep_privacy.txt
python -m tsagent experiment experiments/prompts/sleep_privacy.txt \
    --tools get_daily_sleep,get_night_hypnogram,get_raw_eeg          # also allow raw EEG
python -m tsagent experiment my_prompt.txt --system-file my_system.txt --today 2026-11-06
```

Each run is saved to `experiments/runs/<timestamp>_<name>/`:

| file | contents |
|------|----------|
| `spec.json` | prompt, system prompt, tools, model, date: everything to reproduce the run |
| `trace.jsonl` | every event with its full payload (one JSON object per line) |
| `flow.html` | open in a browser: summary, sequence diagram, expandable timeline |
| `answer.md` | the agent's final answer |

Events recorded:

| event | from → to | payload |
|-------|-----------|---------|
| `prompt` | user → agent | your prompt and the system prompt |
| `llm_request` | agent → llm | the full request (the whole conversation is resent each turn) and which messages are new |
| `llm_response` | llm → agent | text, reasoning (if the model returns it), tool calls |
| `tool_call` | agent → tool | tool name and arguments chosen by the LLM |
| `db_read` / `db_write` | db ↔ tool | DB method, time range, how many rows/samples |
| `tool_result` | tool → agent | exactly what is handed back and then sent to the LLM |
| `answer` | agent → user | the final text |

The summary shows where the LLM runs (local or cloud), how many bytes were sent to it, how much
tool data it saw, and how many raw samples and rows were read from the DB.

## Batch tasks: the agent writes its own database queries

Instead of fixed read functions, the agent gets `query_influxdb`. It writes Flux queries, and the
tool sends them as HTTP requests to InfluxDB's API (`POST /api/v2/query`). The system prompt
describes the database (measurements, fields, time conventions) so the agent knows what to ask for.
It also gets `process_nights`, which it calls when a night has raw EEG but no summary yet. Use
`--tools query_influxdb` to take that away; the agent can then only report that data is missing.

Put one question per `.txt` file in `experiments/tasks/`, then run them all:

```bash
python -m tsagent batch                                   # all tasks in experiments/tasks/
python -m tsagent batch experiments/tasks --today 2026-11-16
docker compose run --rm agent batch                       # same, in the isolated container
```

Results go to `experiments/results/<timestamp>/`:

| file | contents |
|------|----------|
| `summary.json` | one entry per task: question, answer, status, number of queries, duration |
| `<task>.json` | question, answer, every query the agent sent (Flux text, HTTP request, rows returned or error), flow summary, full event trace |
| `<task>.flow.html` | the same flow, for reading in a browser |

The query tool is read-only. Queries that write (`to()`), import packages that can reach other
servers (`http`, `sql`, …), or read another bucket are rejected before anything is sent. Results
are cut at `max_rows` rows (at most 1000), and the HTTP response is streamed and closed at that
point. For real deployments, also give the agent an InfluxDB token that can only read the bucket.
