# Copyright (c) 2025 278097159+leptarip@users.noreply.github.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MLP ensemble fitting and probability helpers for offline classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class NNEnsembleCfg:
    """MLP training and weighted-bootstrap settings."""

    seed: int = 0
    ensemble_size: int = 5
    hidden_layers: Tuple[int, ...] = (64, 64)
    activation: str = "relu"
    alpha: float = 1e-4
    max_iter: int = 400
    batch_size: int = 64
    learning_rate_init: float = 1e-3
    early_stopping: bool = True
    validation_fraction: float = 0.15
    bootstrap: bool = True


def _build_mlp(seed: int, cfg: NNEnsembleCfg) -> Pipeline:
    """
    Build a scaler + MLP pipeline so features are standardized per labeled set.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=cfg.hidden_layers,
                    activation=cfg.activation,
                    alpha=cfg.alpha,
                    max_iter=cfg.max_iter,
                    batch_size=cfg.batch_size,
                    learning_rate_init=cfg.learning_rate_init,
                    solver="adam",
                    early_stopping=cfg.early_stopping,
                    validation_fraction=cfg.validation_fraction,
                    random_state=seed,
                ),
            ),
        ]
    )


def _fit_ensemble(X: np.ndarray, y: np.ndarray, cfg: NNEnsembleCfg) -> List[Pipeline]:
    models: List[Pipeline] = []
    # class weights to handle imbalance (unsafe rare)
    uniq, counts = np.unique(y, return_counts=True)
    weights = {int(cls): float(len(y) / (len(uniq) * cnt)) for cls, cnt in zip(uniq, counts)}
    sample_weights = np.asarray([weights[int(label)] for label in y], dtype=float)
    for i in range(cfg.ensemble_size):
        mdl = _build_mlp(seed=cfg.seed + i, cfg=cfg)
        if cfg.bootstrap:
            rng = np.random.default_rng(cfg.seed + 17 * i)
            p = sample_weights / np.maximum(sample_weights.sum(), 1e-12)
            idx = rng.choice(len(y), size=len(y), replace=True, p=p)
            X_fit = X[idx]
            y_fit = y[idx]
        else:
            X_fit, y_fit = X, y
        # sklearn MLPClassifier.fit does not support sample_weight reliably; rely on weighted bootstrap above.
        mdl.fit(X_fit, y_fit)
        models.append(mdl)
    return models


def _predict_proba_ensemble(models: List[Pipeline], X: np.ndarray) -> np.ndarray:
    if not models:
        return np.zeros((len(X), 2), dtype=float)
    ps = []
    for m in models:
        p = np.asarray(m.predict_proba(X), dtype=float)
        classes = None
        if hasattr(m, "steps"):
            try:
                classes = getattr(m.steps[-1][1], "classes_", None)
            except Exception:
                classes = None
        if classes is None:
            classes = getattr(m, "classes_", None)
        if classes is None:
            if p.ndim == 2 and p.shape[1] == 2:
                p01 = p
            else:
                raise ValueError(f"Cannot infer class mapping for model {type(m)} with predict_proba shape={p.shape}")
        else:
            classes = np.asarray(classes)
            p01 = np.zeros((p.shape[0], 2), dtype=float)
            for j, c in enumerate(classes.tolist()):
                if c in (0, 1):
                    p01[:, int(c)] = p[:, j]
        ps.append(p01)
    probs = np.mean(np.stack(ps, axis=0), axis=0)
    probs = np.clip(probs, 0.0, 1.0)
    row_sum = probs.sum(axis=1, keepdims=True)
    probs = probs / np.maximum(row_sum, 1e-12)
    return probs


def _predict_proba_components(models: List[Pipeline], X: np.ndarray) -> np.ndarray:
    """
    Return per-model unsafe probabilities with consistent class ordering.
    Shape: (n_models, n_samples).
    """
    if not models:
        return np.zeros((0, len(X)), dtype=float)
    comps = []
    for m in models:
        p = np.asarray(m.predict_proba(X), dtype=float)
        classes = None
        if hasattr(m, "steps"):
            try:
                classes = getattr(m.steps[-1][1], "classes_", None)
            except Exception:
                classes = None
        if classes is None:
            classes = getattr(m, "classes_", None)
        if classes is None:
            if p.ndim == 2 and p.shape[1] == 2:
                unsafe = p[:, 0]
            else:
                raise ValueError(f"Cannot infer class mapping for model {type(m)} with predict_proba shape={p.shape}")
        else:
            classes = np.asarray(classes)
            if 0 in classes:
                unsafe = p[:, list(classes).index(0)]
            elif 1 in classes:
                # if only safe class present, treat unsafe prob as 1 - safe
                safe = p[:, list(classes).index(1)]
                unsafe = 1.0 - safe
            else:
                unsafe = p[:, 0]
        comps.append(np.clip(unsafe, 0.0, 1.0))
    return np.vstack(comps)
