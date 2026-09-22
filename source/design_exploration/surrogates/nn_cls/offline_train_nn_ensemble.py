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
"""
Offline training for the NN ensemble surrogate.

Training uses cached episodes only:
- Labels are inferred via offline SPRT on cached episodes.
- Train split fits the ensemble; calibration split (if provided) fits a Platt
  scaling head on top of the ensemble's average probability.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

from source.design_exploration.commons import design_space
from source.design_exploration.surrogates.nn_cls.ensemble import (
    NNEnsembleCfg,
    _fit_ensemble,
    _predict_proba_ensemble,
    _predict_proba_components,
)
from source.design_exploration.commons.scenario_labeling import TestResult
from source.design_exploration.commons.surrogate_storage import (
    default_surrogate_root,
    latest_run_dir,
    prepare_run_dir,
)
from source.design_exploration.commons.traceability import split_trace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
_REPO_ROOT = next((cand for cand in [_HERE] + list(_HERE.parents) if (cand / "source").is_dir()), _HERE.parent)

ALGO_VERSION = "approach1_v1"
EPISODE_ROOT = str(_REPO_ROOT / "episode_dataset")
SURROGATE_ROOT = str(default_surrogate_root(_HERE))
OFFLINE_DATASET_NAME = "offline_builder"
MODEL_OUT = "nn_ensemble_offline.joblib"
TRAINER_VERSION = "v1"
SEED = 13


class CalibratedEnsemble:
    """Lightweight wrapper to attach a 1D calibrator on top of ensemble probs."""

    def __init__(self, models, calibrator=None):
        self.models = models
        self.calibrator = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        base = _predict_proba_ensemble(self.models, X)
        if self.calibrator is None:
            return base
        p_unsafe = base[:, 0].reshape(-1, 1)
        p_unsafe_cal = self.calibrator.transform(p_unsafe.flatten())
        p_unsafe_cal = np.clip(p_unsafe_cal, 0.0, 1.0)
        p_safe_cal = 1.0 - p_unsafe_cal
        return np.stack([p_unsafe_cal, p_safe_cal], axis=1)

    def predict_component_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return per-model unsafe probabilities (uncalibrated), shape (n_models, n_samples).
        Useful for ensemble variance diagnostics.
        """
        return _predict_proba_components(self.models, X)


class TemperatureScaler:
    """Simple temperature scaling on unsafe probability."""

    def __init__(self, T: float = 1.0):
        self.T = float(T)

    def fit(self, p_raw: np.ndarray, y: np.ndarray):
        p = np.clip(p_raw, 1e-6, 1 - 1e-6)
        logits = np.log(p / (1 - p))
        best_T = 1.0
        best_ll = float("inf")
        for T in np.linspace(0.5, 5.0, 30):
            p_cal = 1.0 / (1.0 + np.exp(-logits / T))
            ll = log_loss(y, p_cal, labels=[0, 1])
            if ll < best_ll:
                best_ll = ll
                best_T = T
        self.T = best_T
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        p = np.clip(p_raw, 1e-6, 1 - 1e-6)
        logits = np.log(p / (1 - p))
        p_cal = 1.0 / (1.0 + np.exp(-logits / self.T))
        return p_cal


def _load_labels_map(path: str) -> Dict[int, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(gi): int(lbl) for gi, lbl in dict(data.get("labels", {})).items()}


def _load_split_manifest(split_path: str | None = None, surrogate_root: str | None = None) -> Dict[str, object]:
    path = Path(split_path) if split_path else (_default_dataset_run_dir(surrogate_root) / "offline_splits.json")
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_dataset_run_dir(surrogate_root: str | None = None) -> Path:
    root = surrogate_root or SURROGATE_ROOT
    run_dir = latest_run_dir(root, OFFLINE_DATASET_NAME)
    if run_dir is None:
        raise FileNotFoundError(f"No offline dataset run found under {Path(root) / OFFLINE_DATASET_NAME}")
    return run_dir


def _resolve_space_spec(split_manifest: Dict[str, object], spec_name: str | None = None) -> design_space.SpaceSpec:
    resolved_name = split_manifest.get("space_spec_name") or spec_name
    if not resolved_name:
        raise KeyError("space_spec_name missing from split manifest and not provided explicitly")
    return design_space.get_space_spec(str(resolved_name))


def _collect_labeled(
    grid: np.ndarray,
    indices: Iterable[int],
    labels_map: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    X_rows: List[np.ndarray] = []
    y_labels: List[int] = []
    for gi in indices:
        gi_int = int(gi)
        if gi_int not in labels_map:
            continue
        X_rows.append(grid[gi_int])
        y_labels.append(int(labels_map[gi_int]))
    if not X_rows:
        return np.zeros((0, grid.shape[1]), dtype=float), np.zeros((0,), dtype=int)
    return np.vstack(X_rows), np.asarray(y_labels, dtype=int)


def train_nn_ensemble_offline(
    train_idx: Sequence[int],
    calib_idx: Sequence[int],
    test_idx: Sequence[int] | None = None,
    *,
    space_spec_name: str | None = None,
    episode_root: str | None = None,
    surrogate_root: str | None = None,
    algo_version: str | None = None,
    split_path: str | None = None,
    seed: int | None = None,
    d_safe: float | None = None,
) -> Dict[str, str]:
    episode_root = episode_root or EPISODE_ROOT
    surrogate_root = surrogate_root or SURROGATE_ROOT
    algo_version = algo_version or ALGO_VERSION
    seed = SEED if seed is None else int(seed)
    split_manifest = _load_split_manifest(split_path, surrogate_root)
    scenario_name = split_manifest.get("scenario_name")
    if not scenario_name:
        raise KeyError("scenario_name missing from split manifest")
    labels_path = split_manifest.get("labels_path")
    if not labels_path:
        raise KeyError("labels_path missing from split manifest")
    space_spec = _resolve_space_spec(split_manifest, space_spec_name)
    grid = space_spec.build_grid()
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Labels manifest not found: {labels_path}")
    labels_map = _load_labels_map(labels_path)

    X_train, y_train = _collect_labeled(grid, train_idx, labels_map)
    X_calib, y_calib = _collect_labeled(grid, calib_idx, labels_map)
    X_test, y_test = (np.zeros((0, grid.shape[1])), np.zeros((0,), dtype=int))
    if test_idx is not None:
        X_test, y_test = _collect_labeled(grid, test_idx, labels_map)
    if y_train.size == 0:
        raise RuntimeError("No labeled training designs available from cache.")

    cfg = NNEnsembleCfg(
        seed=seed,
        ensemble_size=5,
        hidden_layers=(128, 64),
        alpha=5e-4,  # weight decay
        max_iter=300,
        batch_size=64,
        learning_rate_init=5e-4,
        early_stopping=True,
        validation_fraction=0.2,
        bootstrap=True,
    )
    models = _fit_ensemble(X_train, y_train, cfg)

    calibrator = None
    if X_calib.size > 0 and np.unique(y_calib).size >= 2:
        try:
            probs_cal_full = _predict_proba_ensemble(models, X_calib)
            # Calibrate on unsafe probability explicitly with temperature scaling.
            p_unsafe_raw = probs_cal_full[:, 0].reshape(-1, 1)
            y_unsafe = (y_calib == int(TestResult.UNSAFE)).astype(int)
            calibrator = TemperatureScaler().fit(p_unsafe_raw.flatten(), y_unsafe)
        except Exception:
            logger.exception("Calibration failed; continuing without calibrator.")
            calibrator = None
    calib_logloss = None
    calib_acc = calib_auc = test_acc = test_auc = None
    if X_calib.size > 0 and np.unique(y_calib).size >= 2:
        try:
            probs_cal = CalibratedEnsemble(models, calibrator).predict_proba(X_calib)
            p_unsafe = probs_cal[:, 0]
            y_unsafe = (y_calib == int(TestResult.UNSAFE)).astype(int)
            calib_logloss = float(log_loss(y_unsafe, np.clip(p_unsafe, 1e-6, 1 - 1e-6), labels=[0, 1]))
            logger.info("[nn_offline] calib logloss=%.4f on %d samples", calib_logloss, len(y_unsafe))
            calib_acc = float(accuracy_score(y_unsafe, (p_unsafe >= 0.5).astype(int)))
            try:
                calib_auc = float(roc_auc_score(y_unsafe, p_unsafe))
            except Exception:
                calib_auc = None
            logger.info("[nn_offline] calib acc=%.4f auc=%s", calib_acc, f"{calib_auc:.4f}" if calib_auc is not None else "n/a")
        except Exception:
            logger.exception("Failed to compute calibration logloss.")
            calib_logloss = None
    test_logloss = None
    if X_test.size > 0 and np.unique(y_test).size >= 2:
        try:
            probs_test = CalibratedEnsemble(models, calibrator).predict_proba(X_test)
            p_unsafe_t = probs_test[:, 0]
            y_unsafe_t = (y_test == int(TestResult.UNSAFE)).astype(int)
            test_logloss = float(log_loss(y_unsafe_t, np.clip(p_unsafe_t, 1e-6, 1 - 1e-6), labels=[0, 1]))
            logger.info("[nn_offline] test logloss=%.4f on %d samples", test_logloss, len(y_unsafe_t))
            test_acc = float(accuracy_score(y_unsafe_t, (p_unsafe_t >= 0.5).astype(int)))
            try:
                test_auc = float(roc_auc_score(y_unsafe_t, p_unsafe_t))
            except Exception:
                test_auc = None
            logger.info("[nn_offline] test acc=%.4f auc=%s", test_acc, f"{test_auc:.4f}" if test_auc is not None else "n/a")
        except Exception:
            logger.exception("Failed to compute test logloss.")
            test_logloss = None

    wrapped = CalibratedEnsemble(models, calibrator)
    out_dir = prepare_run_dir(surrogate_root, "nn_ensemble")
    model_path = os.path.join(out_dir, MODEL_OUT)
    joblib.dump(wrapped, model_path)

    probs_mean = wrapped.predict_proba(grid)
    probs_components = wrapped.predict_component_proba(grid)
    mean_path = os.path.join(out_dir, "nn_probs_mean.npy")
    comp_path = os.path.join(out_dir, "nn_probs_components.npy")
    np.save(mean_path, probs_mean)
    np.save(comp_path, probs_components)

    meta = {
        "scenario": scenario_name,
        "space_spec_name": space_spec.name,
        "space_spec_keys": list(space_spec.keys),
        "episode_root": os.path.join(episode_root, scenario_name),
        "surrogate_root": surrogate_root,
        "run_dir": out_dir,
        "dataset_manifest_path": str(Path(split_path).with_name("offline_artifacts.json")) if split_path else None,
        "labels_path": labels_path,
        "split_path": split_path,
        "trainer_version": TRAINER_VERSION,
        "algo_version": algo_version,
        "model_id": Path(out_dir).name,
        "model_run_id": Path(out_dir).name,
        "model_family": Path(out_dir).parent.name,
        "train_size": int(len(y_train)),
        "calib_size": int(len(y_calib)),
        "test_size": int(len(y_test)),
        "train_idx": [int(x) for x in train_idx],
        "calib_idx": [int(x) for x in calib_idx],
        "test_idx": [int(x) for x in test_idx] if test_idx is not None else [],
        "model_path": model_path,
        "probs_mean": mean_path,
        "probs_components": comp_path,
        "calib_logloss": calib_logloss,
        "test_logloss": test_logloss,
        "calib_accuracy": calib_acc,
        "calib_auc": calib_auc,
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "trace": {
            "model_id": Path(out_dir).name,
            "model_run_id": Path(out_dir).name,
            "model_family": Path(out_dir).parent.name,
            "dataset_manifest_path": str(Path(split_path).with_name("offline_artifacts.json")) if split_path else None,
            "split": split_trace(split_path) if split_path else None,
        },
    }
    meta_path = os.path.join(out_dir, os.path.splitext(MODEL_OUT)[0] + "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("[nn_offline] saved model=%s meta=%s", model_path, meta_path)
    return {
        "model": model_path,
        "meta": meta_path,
        "probs_mean": mean_path,
        "probs_components": comp_path,
    }
