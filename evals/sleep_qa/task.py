"""sleep_qa: how an agent answers health questions from a time-series DB it can only query.

    inspect eval evals/sleep_qa/task.py --model ollama/qwen2.5:7b-instruct-q4_K_M
    inspect eval evals/sleep_qa/task.py -T knowledge=blind --model ...

Each sample gets a fresh Docker sandbox (sandbox/compose.yaml): the agent's tool calls run in an
isolated container that can reach only a query gateway; the gateway alone holds the InfluxDB
token and logs every request for the behaviour scorer.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.agent import AgentPrompt, as_solver, react  # noqa: E402
from inspect_ai.dataset import json_dataset  # noqa: E402
from inspect_ai.solver import system_message  # noqa: E402

from prompts import PROMPTS  # noqa: E402
from scorers import answer_correct, query_behaviour  # noqa: E402
from tools import query_influxdb  # noqa: E402


@task
def sleep_qa(knowledge: str = "informed", max_messages: int = 30):
    if knowledge not in PROMPTS:
        raise ValueError(f"knowledge must be one of {sorted(PROMPTS)}")
    agent = react(
        # The system prompt comes from system_message() so it can use each sample's metadata.
        prompt=AgentPrompt(instructions=None, handoff_prompt=None, assistant_prompt=None, submit_prompt=None),
        tools=[query_influxdb()],
        submit=False,            # the turn ends when the model answers without calling a tool
    )
    return Task(
        dataset=json_dataset(str(HERE / "dataset.jsonl")),
        solver=[system_message(PROMPTS[knowledge]), as_solver(agent)],
        scorer=[answer_correct(), query_behaviour()],
        sandbox=("docker", str(HERE / "sandbox" / "compose.yaml")),
        message_limit=max_messages,
        metadata={"knowledge": knowledge},
    )
