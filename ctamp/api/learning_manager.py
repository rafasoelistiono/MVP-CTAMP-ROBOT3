"""Learning manager for API service."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..learning.heuristic_models import LearnedModel, OfflineSVRModel, OnlineSGDModel
from .models import TrainingRequest, TrainingResult


class LearningManager:
    """Manage learning requests."""

    def __init__(self) -> None:
        self._models: dict[str, LearnedModel] = {}

    def train(self, request: TrainingRequest) -> TrainingResult:
        try:
            if not request.samples:
                return TrainingResult(success=False, error="no_samples")

            X = np.array([s.features for s in request.samples])
            y = np.array([s.cost for s in request.samples])

            model = self._create_model(request.model_type)
            model.fit(X, y)

            self._models[request.model_type] = model

            y_pred = np.array([model.predict(x) for x in X])
            mae = float(np.mean(np.abs(y - y_pred)))
            mse = float(np.mean((y - y_pred) ** 2))

            return TrainingResult(
                success=True,
                num_samples=len(request.samples),
                metrics={"mae": mae, "mse": mse},
            )
        except Exception as e:
            return TrainingResult(success=False, error=str(e))

    def get_model(self, model_type: str) -> Optional[LearnedModel]:
        return self._models.get(model_type)

    def _create_model(self, model_type: str) -> LearnedModel:
        if model_type == "online":
            return OnlineSGDModel()
        return OfflineSVRModel()
