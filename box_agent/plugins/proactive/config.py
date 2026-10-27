"""Opt-in configuration owned by the proactive plugin."""

from pydantic import BaseModel, ConfigDict, Field


class ProactiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    timeout_seconds: float = Field(default=60, ge=1, le=180)
    max_qa_chars: int = Field(default=24000, ge=1000, le=100000)
    evidence_limit: int = Field(default=6, ge=1, le=12)
    recent_limit: int = Field(default=20, ge=1, le=50)
