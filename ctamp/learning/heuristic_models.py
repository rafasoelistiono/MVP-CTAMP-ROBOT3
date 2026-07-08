"""Learned heuristic models for CTAMP."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


EPSILON = 1e-6


class LearnedModel(ABC):
    @abstractmethod
    def predict(self, features: np.ndarray) -> float:
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    def _clip(self, value: float) -> float:
        return max(value, EPSILON)


class ConstantHeuristicModel(LearnedModel):
    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def predict(self, features: np.ndarray) -> float:
        return self._clip(self.value)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(y) > 0:
            self.value = float(np.mean(y))

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.fit(X, y)

    def save(self, path: str) -> None:
        np.save(path, np.array([self.value]))

    def load(self, path: str) -> None:
        self.value = float(np.load(path)[0])


class OfflineSVRModel(LearnedModel):
    def __init__(self, kernel: str = "rbf", C: float = 1.0) -> None:
        self.kernel = kernel
        self.C = C
        self._model = None
        self._fitted = False

    def predict(self, features: np.ndarray) -> float:
        if not self._fitted:
            return self._clip(1.0)
        X = features.reshape(1, -1)
        return self._clip(float(self._model.predict(X)[0]))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.svm import SVR
        self._model = SVR(kernel=self.kernel, C=self.C)
        self._model.fit(X, y)
        self._fitted = True

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if not self._fitted:
            self.fit(X, y)
        else:
            X_all = np.vstack([self._model.support_vectors_, X])
            y_all = np.concatenate([self._model._y if hasattr(self._model, '_y') else y, y])
            self.fit(X_all, y_all)

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "fitted": self._fitted, "kernel": self.kernel, "C": self.C}, f)

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._fitted = data["fitted"]
        self.kernel = data["kernel"]
        self.C = data["C"]


class OnlineSGDModel(LearnedModel):
    def __init__(self, alpha: float = 0.0001) -> None:
        self.alpha = alpha
        self._model = None
        self._fitted = False

    def predict(self, features: np.ndarray) -> float:
        if not self._fitted:
            return self._clip(1.0)
        X = features.reshape(1, -1)
        return self._clip(float(self._model.predict(X)[0]))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import SGDRegressor
        self._model = SGDRegressor(alpha=self.alpha, max_iter=1000, tol=1e-3)
        self._model.fit(X, y)
        self._fitted = True

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import SGDRegressor
        if self._model is None:
            self._model = SGDRegressor(alpha=self.alpha, max_iter=1000, tol=1e-3)
        if not self._fitted:
            self._model.fit(X, y)
            self._fitted = True
        else:
            self._model.partial_fit(X, y)

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "fitted": self._fitted, "alpha": self.alpha}, f)

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._fitted = data["fitted"]
        self.alpha = data["alpha"]
