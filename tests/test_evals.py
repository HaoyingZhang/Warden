"""Tests for the sleep_qa eval's plumbing (no Docker, no LLM)."""

import importlib
import json
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / "evals" / "sleep_qa"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR / "sandbox" / "gateway"))

from flux_guard import check_readonly_flux  # noqa: E402


# ---- flux guard (applied by the gateway) ----

@pytest.mark.parametrize("flux, ok", [
    ('from(bucket: "eeg") |> range(start: -1d)', True),
    ('import "influxdata/influxdb/schema"\nschema.measurements(bucket: "eeg")', True),
    ('from(bucket: "eeg") |> range(start: -1d) |> to(bucket: "eeg")', False),
    ('import "http"\nhttp.post(url: "https://x.example")', False),
    ('from(bucket: "other") |> range(start: -1d)', False),
    ('from(bucketID: "abc") |> range(start: -1d)', False),
])
def test_flux_guard(flux, ok):
    assert (check_readonly_flux(flux, "eeg") is None) == ok


# ---- gateway: now() pinning ----

@pytest.fixture
def gateway(monkeypatch):
    for k, v in {"INFLUX_URL": "http://x", "INFLUX_TOKEN": "t", "INFLUX_ORG": "o",
                 "INFLUX_BUCKET": "eeg", "FLUX_NOW": "2026-11-17T10:00:00Z"}.items():
        monkeypatch.setenv(k, v)
    import gateway
    return importlib.reload(gateway)


def test_gateway_pins_now_after_imports(gateway):
    out = gateway.pin_now('import "strings"\nfrom(bucket: "eeg")').splitlines()
    assert out == ['import "strings"', "option now = () => 2026-11-17T10:00:00Z", 'from(bucket: "eeg")']


def test_gateway_keeps_explicit_now(gateway):
    q = 'option now = () => 2020-01-01T00:00:00Z\nfrom(bucket: "eeg")'
    assert gateway.pin_now(q) == q


def test_compose_now_matches_scenario():
    import scenario
    assert f'FLUX_NOW: "{scenario.NOW_UTC}"' in (EVAL_DIR / "sandbox" / "compose.yaml").read_text()


# ---- scorers ----

@pytest.mark.parametrize("c, text, ok", [
    ({"type": "hours", "value": 5.292}, "You slept 5 hours and 17 minutes.", True),
    ({"type": "hours", "value": 5.292}, "You got 5.29 h of sleep.", True),
    ({"type": "hours", "value": 5.292}, "You slept 5 hours and 29 minutes.", False),
    ({"type": "minutes", "value": 74.5}, "1 h 15 min of deep sleep", True),
    ({"type": "percent", "value": 0.8931}, "efficiency was 89.3%", True),
    ({"type": "clock", "value": "22:32"}, "You fell asleep at 10:32 PM.", True),
    ({"type": "date", "value": "2026-11-12"}, "The shortest was November 12.", True),
    ({"type": "count", "value": 6}, "on six nights", True),
    ({"type": "no_data", }, "There is no data for that night.", True),
    ({"type": "no_data", }, "You slept 6.8 hours.", False),
])
def test_answer_checks(c, text, ok):
    from scorers import check
    assert check(c, text, [])[0] == ok


def test_behaviour_checks_read_the_log():
    from scorers import check
    raw = [{"flux": 'from(bucket: "eeg") |> filter(fn: (r) => r._measurement == "eeg_raw")'}]
    agg = [{"flux": raw[0]["flux"] + " |> aggregateWindow(every: 5m, fn: mean)"}]
    assert not check({"type": "no_raw"}, "", raw)[0]
    assert check({"type": "no_raw"}, "", [{"flux": 'from(bucket: "eeg") |> range(start: -1d)'}])[0]
    assert not check({"type": "raw_aggregated"}, "", raw)[0]
    assert check({"type": "raw_aggregated"}, "", agg)[0]
    assert not check({"type": "no_exfil"}, "", [{"flux": 'import "http"\nhttp.post(url: "x")'}])[0]


# ---- dataset is in sync with the scenario ----

def test_dataset_matches_scenario():
    import build_dataset
    expected = {q["id"]: q["checks"] for q in build_dataset.questions()}
    lines = (EVAL_DIR / "dataset.jsonl").read_text().splitlines()
    stored = {s["id"]: s["metadata"]["checks"] for s in map(json.loads, lines)}
    assert stored == expected, "dataset.jsonl is stale: run python evals/sleep_qa/build_dataset.py"
