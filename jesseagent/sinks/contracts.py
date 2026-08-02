"""Contracts for preview-first systems that apply approved Agent output."""

from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class SinkPlan(BaseModel):
    """An immutable, user-previewable external change proposal."""

    model_config = ConfigDict(frozen=True)

    plan_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    sink_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    preview: str = Field(min_length=1)
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class SinkApplyResult(BaseModel):
    """A compact result safe to persist in the Agent event log."""

    model_config = ConfigDict(frozen=True)

    plan_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class SinkConnector(Protocol):
    """Plan without mutation, then apply only the exact approved plan."""

    sink_id: str

    def plan(self, operation: str, arguments: dict[str, JsonValue]) -> SinkPlan:
        """Return a stable change plan and human-readable preview."""

    def apply(self, approved_plan: SinkPlan) -> SinkApplyResult:
        """Apply an already-approved plan without replanning it."""
