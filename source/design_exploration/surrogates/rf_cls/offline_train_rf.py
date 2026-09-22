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
Offline RF surrogate training from cached episodes.

The training function accepts two split index arrays (train, calib). It fits a
RandomForest on the train split, then calibrates on the calib split (if
available) and persists the model.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score, accuracy_score

from source.design_exploration.commons import design_space
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

# Defaults (override via CLI)
ALGO_VERSION = "approach1_v1"
EPISODE_ROOT = str(_REPO_ROOT / "episode_dataset")
SURROGATE_ROOT = str(default_surrogate_root(_HERE))
OFFLINE_DATASET_NAME = "offline_builder"
MODEL_OUT = "rf_offline.joblib"
TRAINER_VERSION = "v1"
SEED = 13
# RF search defaults: narrowed to 6 configs around the current best region,
# plus 2 exploratory variants that may improve test-time calibration.
RF_SEARCH_CONFIGS = (
    {"min_samples_leaf": 30, "max_depth": 16, "n_estimators": 200, "max_features": 1.0},
    {"min_samples_leaf": 30, "max_depth": 12, "n_estimators": 200, "max_features": 1.0},
    {"min_samples_leaf": 20, "max_depth": 16, "n_estimators": 200, "max_features": 0.8},
    {"min_samples_leaf": 20, "max_depth": 12, "n_estimators": 200, "max_features": 0.8},
    {"min_samples_leaf": 40, "max_depth": 16, "n_estimators": 300, "max_features": 1.0},
    {"min_samples_leaf": 25, "max_depth": None, "n_estimators": 300, "max_features": 0.8},
)
RF_CLASS_WEIGHT = "balanced"
# Probability calibration map fitted on the calibration split. Isotonic produced
# the published scores and the audited set, so it is the default here. It is a
# step function: it maps large blocks of designs onto identical probabilities,
# including exactly 0 and 1, and a threshold cannot be placed inside such a
# block. "sigmoid" (Platt) is strictly monotone and keeps the forest's full score
# resolution where finer thresholds are needed.
RF_CALIBRATION_METHOD = "isotonic"


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


def _resolve_space_spec(split_manifest: Dict[str, object]) -> design_space.SpaceSpec:
    spec_name = split_manifest.get("space_spec_name")
    if not spec_name:
        raise KeyError("space_spec_name missing from split manifest")
    return design_space.get_space_spec(str(spec_name))


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


def train_rf_offline(
    train_idx: Sequence[int],
    calib_idx: Sequence[int],
    test_idx: Sequence[int] | None = None,
    *,
    episode_root: str | None = None,
    surrogate_root: str | None = None,
    algo_version: str | None = None,
    split_path: str | None = None,
    seed: int | None = None,
    rf_class_weight: str | None = None,
    d_safe: float | None = None,
) -> Dict[str, str]:
    episode_root = episode_root or EPISODE_ROOT
    surrogate_root = surrogate_root or SURROGATE_ROOT
    algo_version = algo_version or ALGO_VERSION
    seed = SEED if seed is None else int(seed)
    rf_class_weight = rf_class_weight or RF_CLASS_WEIGHT
    split_manifest = _load_split_manifest(split_path, surrogate_root)
    scenario_name = split_manifest.get("scenario_name")
    if not scenario_name:
        raise KeyError("scenario_name missing from split manifest")
    labels_path = split_manifest.get("labels_path")
    if not labels_path:
        raise KeyError("labels_path missing from split manifest")
    space_spec = _resolve_space_spec(split_manifest)
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

    uniq = np.unique(y_train)
    search_results: List[Dict] = []
    best_model = None
    best_cfg = None
    best_ll = float("inf")
    if uniq.size < 2:
        logger.warning("Only one class in training labels; using DummyClassifier.")
        model = DummyClassifier(strategy="prior")
        model.fit(X_train, y_train)
        best_model = model
        best_cfg = {"type": "dummy"}
    else:
        combos: List[Tuple[int, int | None, int, object]] = [
            (
                int(cfg["min_samples_leaf"]),
                None if cfg["max_depth"] in [None, "None"] else int(cfg["max_depth"]),
                int(cfg["n_estimators"]),
                cfg["max_features"],
            )
            for cfg in RF_SEARCH_CONFIGS
        ]

        if X_calib.size == 0 or np.unique(y_calib).size < 2:
            logger.warning("Calibration set missing or single-class; training first RF config only.")
            combos = combos[:1]

        for min_leaf, max_depth, n_est, max_feat in combos:
            base_model = RandomForestClassifier(
                n_estimators=int(n_est),
                random_state=seed,
                class_weight=rf_class_weight,
                min_samples_leaf=int(min_leaf),
                max_depth=None if max_depth in [None, "None"] else int(max_depth) if max_depth is not None else None,
                max_features=max_feat,
                n_jobs=-1,
            )
            base_model.fit(X_train, y_train)
            model_candidate = base_model
            ll_val = None
            calib_used = False
            if X_calib.size > 0 and np.unique(y_calib).size >= 2:
                try:
                    try:
                        calib = CalibratedClassifierCV(estimator=base_model, method=RF_CALIBRATION_METHOD, cv="prefit")
                    except TypeError:
                        calib = CalibratedClassifierCV(base_estimator=base_model, method=RF_CALIBRATION_METHOD, cv="prefit")
                    calib.fit(X_calib, y_calib)
                    model_candidate = calib
                    calib_used = True
                except Exception:
                    logger.exception("Calibration failed for RF(min_leaf=%s, max_depth=%s); using uncalibrated model.", min_leaf, max_depth)
                    model_candidate = base_model
            try:
                # map column for class label 1 (SAFE)
                probs = model_candidate.predict_proba(X_calib if X_calib.size else X_train)
                classes = getattr(model_candidate, "classes_", None)
                col = 1
                if classes is not None:
                    try:
                        col = list(classes).index(1)
                    except ValueError:
                        col = 1 if probs.shape[1] > 1 else 0
                p_use = probs[:, col]
                target = y_calib if X_calib.size else y_train
                ll_val = float(log_loss(target, p_use, labels=[0, 1]))
            except Exception:
                ll_val = float("inf")

            search_results.append(
                dict(
                    min_samples_leaf=int(min_leaf),
                    max_depth=None if max_depth in [None, "None"] else max_depth,
                    n_estimators=int(n_est),
                    max_features=max_feat,
                    calibrated=calib_used,
                    log_loss=ll_val,
                )
            )
            if ll_val is not None and ll_val < best_ll:
                best_ll = ll_val
                best_cfg = dict(
                    min_samples_leaf=int(min_leaf),
                    max_depth=None if max_depth in [None, "None"] else max_depth,
                    n_estimators=int(n_est),
                    max_features=max_feat,
                    calibrated=calib_used,
                )
                best_model = model_candidate

        if best_model is None:
            best_model = base_model
            best_cfg = {"min_samples_leaf": int(combos[0][0]), "max_depth": combos[0][1], "calibrated": False}
        logger.info("[rf_offline] tried %d RF configs, best logloss=%.4f cfg=%s", len(search_results), best_ll, best_cfg)
        for res in sorted(search_results, key=lambda r: r.get("log_loss", float("inf"))):
            logger.info(
                "[rf_offline][candidate] leaf=%s depth=%s n_estimators=%s max_features=%s calibrated=%s logloss=%.4f",
                res["min_samples_leaf"],
                res["max_depth"],
                res.get("n_estimators"),
                res.get("max_features"),
                res["calibrated"],
                res["log_loss"],
            )

    out_dir = prepare_run_dir(surrogate_root, "rf")
    model_path = os.path.join(out_dir, MODEL_OUT)
    joblib.dump(best_model, model_path)

    # Test logloss if test set available
    test_logloss = None
    calib_acc = calib_auc = test_acc = test_auc = None
    if X_calib.size > 0 and np.unique(y_calib).size >= 2:
        try:
            probs_cal = best_model.predict_proba(X_calib)
            classes = getattr(best_model, "classes_", None)
            col = 1
            if classes is not None:
                try:
                    col = list(classes).index(1)
                except ValueError:
                    col = 1 if probs_cal.shape[1] > 1 else 0
            p_cal = np.clip(probs_cal[:, col], 1e-6, 1 - 1e-6)
            calib_acc = float(accuracy_score(y_calib, (p_cal >= 0.5).astype(int)))
            try:
                calib_auc = float(roc_auc_score(y_calib, p_cal))
            except Exception:
                calib_auc = None
            logger.info("[rf_offline] calib acc=%.4f auc=%s", calib_acc, f"{calib_auc:.4f}" if calib_auc is not None else "n/a")
        except Exception:
            logger.exception("Failed to compute RF calib metrics.")
    else:
        logger.info(
            "[rf_offline] skipping calib metrics (size=%d, classes=%s)",
            len(y_calib),
            np.unique(y_calib).tolist() if y_calib.size else [],
        )

    if X_test.size > 0 and np.unique(y_test).size >= 2:
        try:
            probs_test = best_model.predict_proba(X_test)
            classes = getattr(best_model, "classes_", None)
            col = 1
            if classes is not None:
                try:
                    col = list(classes).index(1)
                except ValueError:
                    col = 1 if probs_test.shape[1] > 1 else 0
            p_use = np.clip(probs_test[:, col], 1e-6, 1 - 1e-6)
            test_logloss = float(log_loss(y_test, p_use, labels=[0, 1]))
            logger.info("[rf_offline] test logloss=%.4f on %d samples", test_logloss, len(y_test))
            test_acc = float(accuracy_score(y_test, (p_use >= 0.5).astype(int)))
            try:
                test_auc = float(roc_auc_score(y_test, p_use))
            except Exception:
                test_auc = None
            logger.info("[rf_offline] test acc=%.4f auc=%s", test_acc, f"{test_auc:.4f}" if test_auc is not None else "n/a")
        except Exception:
            logger.exception("Failed to compute RF test logloss.")
    else:
        logger.info(
            "[rf_offline] skipping test metrics (size=%d, classes=%s)",
            len(y_test),
            np.unique(y_test).tolist() if y_test.size else [],
        )

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
        "rf_search": search_results,
        "best_cfg": best_cfg,
        "best_logloss": None if best_ll == float("inf") else float(best_ll),
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
    logger.info("[rf_offline] saved best model=%s meta=%s", model_path, meta_path)
    return {"model": model_path, "meta": meta_path}
