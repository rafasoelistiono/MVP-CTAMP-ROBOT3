from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from task_planning.types import Step


class RecoveryAction(str, Enum):
    RETRY = "retry"
    REPLAN_REQUIRED = "replan_required"
    ABORT = "abort"


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str


class RecoveryPolicy:
    """Bounded recovery; it never calls an LLM or relaxes safety."""

    _FATAL_TOKENS = (
        "obstacle_displaced",
        "obstacle_fallen",
        "fragile",
        "collision_at_waypoint",
    )

    def __init__(self, max_retries_per_object: int):
        self.max_retries_per_object = max(0, int(max_retries_per_object))

    def decide(
        self,
        step: Step,
        attempt: int,
        failure_reason: str,
        *,
        object_still_held: bool,
    ) -> RecoveryDecision:
        normalized = failure_reason.lower()
        if any(token in normalized for token in self._FATAL_TOKENS):
            return RecoveryDecision(RecoveryAction.ABORT, failure_reason)
        if step.action == "pick" and attempt <= self.max_retries_per_object:
            return RecoveryDecision(RecoveryAction.RETRY, failure_reason)
        if step.action == "place" and object_still_held:
            if attempt <= self.max_retries_per_object:
                return RecoveryDecision(RecoveryAction.RETRY, failure_reason)
        if step.action == "place":
            return RecoveryDecision(RecoveryAction.REPLAN_REQUIRED, failure_reason)
        return RecoveryDecision(RecoveryAction.ABORT, failure_reason)
