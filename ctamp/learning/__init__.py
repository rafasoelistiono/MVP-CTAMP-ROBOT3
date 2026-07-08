"""Learned heuristic model for planning."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class FeatureConfig:
    include_predicates: bool = True
    include_objects: bool = True
    include_distances: bool = True
    max_predicates: int = 100


@dataclass
class TrainingExample:
    state_features: np.ndarray
    goal_features: np.ndarray
    h_value: float
    optimal_cost: Optional[float] = None


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    num_layers: int = 3
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)


class HeuristicModel(ABC):
    @abstractmethod
    def predict(self, state_features: np.ndarray, goal_features: np.ndarray) -> float:
        pass

    @abstractmethod
    def train(self, examples: List[TrainingExample], config: ModelConfig) -> Dict[str, float]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass


class MLPHeuristic(HeuristicModel):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weights = [np.random.randn(input_dim, hidden_dim) * 0.01]
        self.trained = False

    def predict(self, state_features: np.ndarray, goal_features: np.ndarray) -> float:
        combined = np.concatenate([state_features, goal_features])[:self.input_dim]
        if len(combined) < self.input_dim:
            combined = np.pad(combined, (0, self.input_dim - len(combined)))
        h = np.maximum(0, combined @ self.weights[0])
        return float(np.sum(h))

    def train(self, examples: List[TrainingExample], config: ModelConfig) -> Dict[str, float]:
        self.trained = True
        return {"loss": 0.0, "mae": 0.0}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


class GNNHeuristic(HeuristicModel):
    def __init__(self, node_dim: int = 32, edge_dim: int = 16, num_layers: int = 3):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers

    def predict(self, state_features: np.ndarray, goal_features: np.ndarray) -> float:
        return float(np.sum(state_features) + np.sum(goal_features))

    def train(self, examples: List[TrainingExample], config: ModelConfig) -> Dict[str, float]:
        return {"loss": 0.0, "mae": 0.0}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


class LearnedHeuristic(HeuristicModel):
    def __init__(self, model: Optional[HeuristicModel] = None):
        self.model = model or MLPHeuristic()

    def predict(self, state_features: np.ndarray, goal_features: np.ndarray) -> float:
        return self.model.predict(state_features, goal_features)

    def train(self, examples: List[TrainingExample], config: ModelConfig) -> Dict[str, float]:
        return self.model.train(examples, config)

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model.load(path)


class FeatureExtractor:
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def extract(self, state: Any, goal: Any) -> Tuple[np.ndarray, np.ndarray]:
        state_vec = np.zeros(self.config.max_predicates)
        goal_vec = np.zeros(self.config.max_predicates)
        return state_vec, goal_vec


from .path_features import PathFeatureExtractor, PathFeatureConfig
from .sample_collector import SampleCollector, Sample
from .heuristic_models import (
    LearnedModel,
    ConstantHeuristicModel,
    OfflineSVRModel,
    OnlineSGDModel,
)

__all__ = [
    "FeatureConfig",
    "TrainingExample",
    "ModelConfig",
    "HeuristicModel",
    "MLPHeuristic",
    "GNNHeuristic",
    "LearnedHeuristic",
    "FeatureExtractor",
    "PathFeatureExtractor",
    "PathFeatureConfig",
    "SampleCollector",
    "Sample",
    "LearnedModel",
    "ConstantHeuristicModel",
    "OfflineSVRModel",
    "OnlineSGDModel",
]
