"""Settings loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()


def _local_tz_name() -> str:
    try:
        link = os.path.realpath("/etc/localtime")
        if "zoneinfo/" in link:
            return link.split("zoneinfo/", 1)[1]
    except OSError:
        pass
    return "UTC"


@dataclass(frozen=True)
class Settings:
    influx_url: str = os.getenv("INFLUX_URL", "http://localhost:8086")
    influx_token: str = os.getenv("INFLUX_TOKEN", "dev-token-change-me")
    influx_org: str = os.getenv("INFLUX_ORG", "tsagent")
    influx_bucket: str = os.getenv("INFLUX_BUCKET", "eeg")

    subject: str = os.getenv("TSAGENT_SUBJECT", "me")
    channel: str = os.getenv("TSAGENT_CHANNEL", "EEG")
    tz_name: str = os.getenv("TSAGENT_TZ", _local_tz_name())
    stager: str = os.getenv("TSAGENT_STAGER", "yasa")

    # LLM backend: "ollama" (local model) or "anthropic" (Claude API, needs a paid API key).
    llm: str = os.getenv("TSAGENT_LLM", "ollama")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    model: str = os.getenv("TSAGENT_MODEL", "claude-opus-5-5")  # used when llm == "anthropic"

    @property
    def llm_model(self) -> str:
        return self.ollama_model if self.llm == "ollama" else self.model

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.tz_name)


settings = Settings()
