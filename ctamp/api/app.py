"""FastAPI application for CTAMP planning service."""

from __future__ import annotations

from fastapi import FastAPI

from .learning_manager import LearningManager
from .models import (
    HealthResponse,
    PlanningProblemRequest,
    PlanningResult,
    TrainingRequest,
    TrainingResult,
)
from .planning_manager import PlanningManager


class AppState:
    planning: PlanningManager = PlanningManager()
    learning: LearningManager = LearningManager()


state = AppState()

app = FastAPI(
    title="CTAMP Planning Service",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/planning/run", response_model=PlanningResult)
def planning_run(request: PlanningProblemRequest) -> PlanningResult:
    return state.planning.run(request)


@app.post("/learning/train", response_model=TrainingResult)
def learning_train(request: TrainingRequest) -> TrainingResult:
    return state.learning.train(request)
