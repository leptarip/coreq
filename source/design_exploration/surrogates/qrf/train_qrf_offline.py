#!/usr/bin/env python3
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
Offline Quantile Random Forest (QRF) training from cached episodes.

This script is intended to run in the Python 3.11+ environment that has
`quantile_forest` installed. It reads cached episodes, computes the safety
margin target g = min_d - d_safe, then trains a RandomForestQuantileRegressor
and saves the model + metadata.

This module is used programmatically by the offline builder and subprocess trainer.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import joblib
from joblib import dump
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression

# Ensure repository root (folder containing "source") is on sys.path.
_HERE = Path(__file__).resolve()
for cand in [_HERE] + list(_HERE.parents):
    if (cand / "source").is_dir() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
        break

from source.design_exploration.commons import design_space
from source.design_exploration.commons.design_space import SpaceSpec
from source.design_exploration.commons.episodes.episode_manager import EpisodeManager
from source.design_exploration.commons.episodes.episode_result import EpisodeResult
from source.design_exploration.commons.surrogate_storage import (
    default_surrogate_root,
    latest_run_dir,
    prepare_run_dir,
)
from source.design_exploration.commons.traceability import split_trace
from source.design_exploration.surrogates.hetGP.heteroscedastic_gp import _margin_from_episode

LOG = logging.getLogger("ipc.train_qrf_offline")

ALGO_VERSION = "v1"
EPISODE_ROOT = str(next((cand for cand in [_HERE] + list(_HERE.parents) if (cand / "source").is_dir()), _HERE.parent) / "episode_dataset")
SURROGATE_ROOT = str(default_surrogate_root(_HERE))
OFFLINE_DATASET_NAME = "offline_builder"
MODEL_OUT = "qrf_offline.joblib"
TRAINER_VERSION = "v1"
SEED = 13
D_SAFE = 0.5
MIN_EPS_PER_DESIGN = 30
MAX_EPS_PER_DESIGN = 60
QRF_N_ESTIMATORS = 1500
QRF_MIN_SAMPLES_LEAF = 3
EVAL_QUANTILE = 0.5
CALIB_QUANTILES = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.25, 0.5, 0.75]
QRF_CDF_N = 999
P_CAL_UPPER_ENVELOPE = True
# Disable heuristic tail inflation: it creates a probability floor and a
# downward jump at p=0.1. Safety certification belongs to the independent audit.
# Retain these metadata fields so historical artifacts remain reproducible.
P_CAL_TAIL_ALPHA = 1.0
P_CAL_TAIL_POWER = 1.0


class CalibratedQRF:
    def __init__(self, model: RandomForestQuantileRegressor, calib_deltas: Dict[float, float]):
        self.model = model
        self.calib_deltas = {float(k): float(v) for k, v in calib_deltas.items()}
        self.calib_quantiles = sorted(self.calib_deltas.keys())

    def _delta_for_q(self, q: float) -> float:
        if not self.calib_quantiles:
            return 0.0
        if q in self.calib_deltas:
            return float(self.calib_deltas[q])
        qs = np.asarray(self.calib_quantiles, dtype=float)
        ds = np.asarray([self.calib_deltas[qq] for qq in qs], dtype=float)
        if q <= qs[0]:
            return float(ds[0])
        if q >= qs[-1]:
            return float(ds[-1])
        idx = int(np.searchsorted(qs, q))
        q0, q1 = float(qs[idx - 1]), float(qs[idx])
        d0, d1 = float(ds[idx - 1]), float(ds[idx])
        if q1 == q0:
            return d0
        t = (q - q0) / (q1 - q0)
        return float(d0 + t * (d1 - d0))

    def predict(self, X: np.ndarray, quantiles=None):
        if quantiles is None:
            return self.model.predict(X)
        qs_arr = np.atleast_1d(np.asarray(quantiles, dtype=float)).reshape(-1)
        qs_list = [float(q) for q in qs_arr.tolist()]
        if len(qs_list) == 1:
            preds = self.model.predict(X, quantiles=qs_list[0])
            return np.asarray(preds, dtype=float) + self._delta_for_q(qs_list[0])
        preds = self.model.predict(X, quantiles=qs_list)
        deltas = np.array([self._delta_for_q(float(q)) for q in qs_list], dtype=float)
        if preds.ndim == 1:
            return preds + deltas[0]
        return preds + deltas.reshape(1, -1)


def _payload_to_episode_result(payload_dict: dict) -> EpisodeResult:
    ep = EpisodeResult()
    ep.result = payload_dict if isinstance(payload_dict, dict) else {"payload": payload_dict}
    return ep


def _find_episode_root(episode_root: Path, scenario_name: str) -> Path:
    ep_root = episode_root / scenario_name
    if not ep_root.is_dir():
        raise FileNotFoundError(f"Episode directory not found: {ep_root}")
    return ep_root


def _default_dataset_run_dir(surrogate_root: str | None = None) -> Path:
    root = surrogate_root or SURROGATE_ROOT
    run_dir = latest_run_dir(root, OFFLINE_DATASET_NAME)
    if run_dir is None:
        raise FileNotFoundError(f"No offline dataset run found under {Path(root) / OFFLINE_DATASET_NAME}")
    return run_dir


def _load_split_manifest(split_path: str | None = None, surrogate_root: str | None = None) -> Dict[str, object]:
    path = Path(split_path) if split_path else (_default_dataset_run_dir(surrogate_root) / "offline_splits.json")
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_space_spec(split_manifest: Dict[str, object], spec_name: str | None = None) -> SpaceSpec:
    resolved_name = split_manifest.get("space_spec_name") or spec_name
    if not resolved_name:
        raise KeyError("space_spec_name missing from split manifest and not provided explicitly")
    return design_space.get_space_spec(str(resolved_name))


def _normalize_grid(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g_min = grid.min(axis=0)
    g_max = grid.max(axis=0)
    g_range = np.where(g_max > g_min, g_max - g_min, 1.0)
    return (grid - g_min) / g_range, g_min, g_max, g_range


def _build_episode_dataset(
    epman: EpisodeManager,
    grid: np.ndarray,
    allowed_idx: Sequence[int],
    max_eps: int,
    min_eps: int,
    d_safe: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    grid_norm, g_min, g_max, g_range = _normalize_grid(grid)
    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    design_ids: List[int] = []
    weights: List[float] = []
    used_designs = 0
    skipped_designs = 0
    eps_counts: List[int] = []
    for gi in allowed_idx:
        eps = epman.episodes(int(gi), consume=False, limit=max_eps)
        vals = []
        for row in eps:
            payload = row.get("payload", {})
            m = _margin_from_episode(_payload_to_episode_result(payload), d_safe)
            if m is None:
                continue
            if not np.isfinite(m):
                continue
            vals.append(float(m))
        if len(vals) < min_eps:
            skipped_designs += 1
            continue
        used_designs += 1
        x = grid_norm[int(gi)]
        vals_use = vals
        eps_counts.append(len(vals_use))
        w = 1.0 / float(len(vals_use)) if len(vals_use) > 0 else 1.0
        for m in vals_use:
            X_rows.append(x)
            y_rows.append(float(m))
            design_ids.append(int(gi))
            weights.append(float(w))
    if not X_rows:
        return (
            np.zeros((0, grid.shape[1]), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=float),
            {
                "total_designs": int(len(allowed_idx)),
                "used_designs": 0,
                "skipped_designs": int(len(allowed_idx)),
                "total_rows": 0,
            },
        )
    X = np.vstack(X_rows)
    y = np.asarray(y_rows, dtype=float)
    d = np.asarray(design_ids, dtype=int)
    w = np.asarray(weights, dtype=float)
    # Normalize weights so the mean weight is 1.0 (scale invariance for splits).
    w_sum = float(np.sum(w))
    if w_sum > 0:
        w = w * (len(w) / w_sum)
    stats = {
        "total_designs": int(len(allowed_idx)),
        "used_designs": int(used_designs),
        "skipped_designs": int(skipped_designs),
        "total_rows": int(len(y_rows)),
        "sampling": "weights_per_design",
        "weight_norm": {
            "sum": float(w_sum),
            "mean": float(np.mean(w)) if w.size else 0.0,
        },
        "episodes_per_design": {
            "min": int(np.min(eps_counts)) if eps_counts else 0,
            "max": int(np.max(eps_counts)) if eps_counts else 0,
            "mean": float(np.mean(eps_counts)) if eps_counts else 0.0,
        },
        "norm": {
            "g_min": g_min.tolist(),
            "g_max": g_max.tolist(),
            "g_range": g_range.tolist(),
        },
    }
    return X, y, d, w, stats


def _eval_metrics(
    model: RandomForestQuantileRegressor | CalibratedQRF,
    X: np.ndarray,
    y: np.ndarray,
    quantile: float = EVAL_QUANTILE,
) -> Tuple[Optional[float], Optional[float]]:
    if y.size == 0:
        return None, None
    preds = model.predict(X, quantiles=quantile).reshape(-1)
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    mae = float(np.mean(np.abs(preds - y)))
    return rmse, mae


def _calibrate_quantiles(
    model: RandomForestQuantileRegressor,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    quantiles: Sequence[float],
) -> Tuple[Dict[float, float], Dict[float, float], Dict[float, float]]:
    qs = [float(q) for q in quantiles]
    if y_calib.size == 0:
        return {}, {}, {}
    preds = model.predict(X_calib, quantiles=qs)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    deltas: Dict[float, float] = {}
    cov_pre: Dict[float, float] = {}
    cov_post: Dict[float, float] = {}
    for i, q in enumerate(qs):
        pred_q = preds[:, i]
        residuals = y_calib - pred_q
        delta = float(np.quantile(residuals, q))
        deltas[q] = delta
        cov_pre[q] = float(np.mean(y_calib <= pred_q))
        cov_post[q] = float(np.mean(y_calib <= pred_q + delta))
    return deltas, cov_pre, cov_post


def _predict_p_unsafe(model: RandomForestQuantileRegressor | CalibratedQRF, X: np.ndarray, taus: Sequence[float]) -> np.ndarray:
    qs = np.asarray(taus, dtype=float).reshape(-1)
    qs = np.sort(np.unique(qs))
    qs_list = [float(q) for q in qs.tolist()]
    preds = np.asarray(model.predict(X, quantiles=qs_list), dtype=float)
    n_samples = X.shape[0]
    n_q = qs.size
    if preds.ndim == 1:
        if n_samples == 1 and preds.shape[0] == n_q:
            preds = preds.reshape(1, n_q)
        elif preds.shape[0] == n_samples:
            preds = preds.reshape(n_samples, 1)
        else:
            raise ValueError(f"Unexpected QRF prediction shape {preds.shape} for {n_samples} samples")
    if preds.ndim == 2:
        if preds.shape == (n_q, n_samples):
            preds = preds.T
        elif preds.shape != (n_samples, n_q) and not (n_samples == 1 and preds.shape[1] == n_q):
            raise ValueError(f"Unexpected QRF prediction shape {preds.shape} for {n_samples} samples and {n_q} qs")
    preds = np.maximum.accumulate(preds, axis=1)
    mask_ge = preds >= 0.0
    any_ge = mask_ge.any(axis=1)
    p = np.full(n_samples, 1.0, dtype=float)
    if np.any(any_ge):
        idx_hi = np.argmax(mask_ge, axis=1)
        rows_any = np.where(any_ge)[0]
        hi = idx_hi[rows_any]
        hi0 = hi == 0
        if np.any(hi0):
            p[rows_any[hi0]] = qs[0]
        sel = hi > 0
        if np.any(sel):
            rows = rows_any[sel]
            hi_sel = hi[sel]
            lo_sel = hi_sel - 1
            q_lo = preds[rows, lo_sel]
            q_hi = preds[rows, hi_sel]
            t_lo = qs[lo_sel]
            t_hi = qs[hi_sel]
            denom = q_hi - q_lo
            frac = np.where(denom > 1e-12, (0.0 - q_lo) / denom, 1.0)
            frac = np.clip(frac, 0.0, 1.0)
            p_val = t_lo + frac * (t_hi - t_lo)
            p[rows] = np.clip(p_val, t_lo, t_hi)
    return np.clip(p, 0.0, 1.0)


def _eval_classification_metrics(
    model: RandomForestQuantileRegressor | CalibratedQRF,
    X: np.ndarray,
    y_margin: np.ndarray,
    taus: Sequence[float],
    p_calibration: Optional[Dict[str, object]] = None,
) -> Dict[str, Optional[float]]:
    if y_margin.size == 0:
        return {"logloss": None, "accuracy": None, "auc": None}
    y_bin = (y_margin < 0.0).astype(int)
    p = _predict_p_unsafe(model, X, taus)
    p = _apply_p_calibration(p, p_calibration)
    mask = np.isfinite(p)
    if not np.any(mask):
        return {"logloss": None, "accuracy": None, "auc": None}
    y_bin = y_bin[mask]
    p = p[mask]
    metrics: Dict[str, Optional[float]] = {}
    if np.unique(y_bin).size < 2:
        metrics["logloss"] = None
        metrics["auc"] = None
    else:
        metrics["logloss"] = float(log_loss(y_bin, p, labels=[0, 1]))
        metrics["auc"] = float(roc_auc_score(y_bin, p))
    pred_label = (p >= 0.5).astype(int)
    metrics["accuracy"] = float(accuracy_score(y_bin, pred_label))
    return metrics


def _unsafe_rate(y_margin: np.ndarray) -> Optional[float]:
    if y_margin.size == 0:
        return None
    return float(np.mean(y_margin < 0.0))


def _apply_p_calibration(p: np.ndarray, calib: Optional[Dict[str, object]]) -> np.ndarray:
    if calib is None:
        return p
    xs = np.asarray(calib.get("x", []), dtype=float)
    ys = np.asarray(calib.get("y", []), dtype=float)
    if xs.size == 0 or ys.size == 0:
        return p
    p_cal = np.interp(p, xs, ys, left=ys[0], right=ys[-1])
    tail_alpha = float(calib.get("tail_alpha", 1.0))
    tail_power = float(calib.get("tail_power", 1.0))
    if tail_alpha > 1.0 or tail_power > 1.0:
        p_cal = np.where(p_cal < 0.1, np.minimum(1.0, tail_alpha * np.power(p_cal, 1.0 / tail_power)), p_cal)
    if bool(calib.get("upper_envelope", True)):
        p_cal = np.maximum(p_cal, p)
    return np.clip(p_cal, 0.0, 1.0)


def _fit_p_calibrator(
    model: RandomForestQuantileRegressor | CalibratedQRF,
    X_all: np.ndarray,
    y_all: np.ndarray,
    design_ids: np.ndarray,
    calib_mask: np.ndarray,
    taus: Sequence[float],
) -> Optional[Dict[str, object]]:
    if y_all.size == 0 or not np.any(calib_mask):
        return None
    counts: Dict[int, int] = {}
    unsafe: Dict[int, int] = {}
    for di, y in zip(design_ids[calib_mask], y_all[calib_mask]):
        di_int = int(di)
        counts[di_int] = counts.get(di_int, 0) + 1
        if y < 0.0:
            unsafe[di_int] = unsafe.get(di_int, 0) + 1
    if not counts:
        return None
    design_list = sorted(counts.keys())
    rates = np.array([unsafe.get(di, 0) / counts[di] for di in design_list], dtype=float)
    if np.unique(rates).size < 2:
        return None
    X_map: Dict[int, np.ndarray] = {}
    for di, x in zip(design_ids, X_all):
        di_int = int(di)
        if di_int not in X_map:
            X_map[di_int] = x
    X_design = np.vstack([X_map[di] for di in design_list])
    p_raw = _predict_p_unsafe(model, X_design, taus)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw, rates)
    return {
        "x": iso.X_thresholds_.tolist(),
        "y": iso.y_thresholds_.tolist(),
        "upper_envelope": bool(P_CAL_UPPER_ENVELOPE),
        "tail_alpha": float(P_CAL_TAIL_ALPHA),
        "tail_power": float(P_CAL_TAIL_POWER),
    }


def train_qrf_offline(
    train_idx: Sequence[int],
    calib_idx: Sequence[int],
    test_idx: Optional[Sequence[int]] = None,
    *,
    space_spec_name: str | None = None,
    episode_root: str | None = None,
    surrogate_root: str | None = None,
    algo_version: str | None = None,
    split_path: str | None = None,
    seed: int | None = None,
    d_safe: float | None = None,
    out_dir: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Train a QRF surrogate using the provided splits (same target as other regressors).
    """
    episode_root = episode_root or EPISODE_ROOT
    surrogate_root = surrogate_root or SURROGATE_ROOT
    algo_version = algo_version or ALGO_VERSION
    seed = SEED if seed is None else int(seed)
    d_safe = D_SAFE if d_safe is None else float(d_safe)
    split_manifest = _load_split_manifest(split_path, surrogate_root)
    scenario_name = split_manifest.get("scenario_name")
    if not scenario_name:
        raise KeyError("scenario_name missing from split manifest")
    space_spec = _resolve_space_spec(split_manifest, space_spec_name)
    grid = space_spec.build_grid()
    ep_root = _find_episode_root(Path(episode_root), scenario_name)
    epman = EpisodeManager(
        root_dir=str(ep_root),
        scenario_name=str(scenario_name),
        space_spec=space_spec,
        GRID=grid,
        flush_every=50,
    )

    t_start = time.time()
    allowed_union = np.unique(
        np.asarray(list(train_idx) + list(calib_idx) + ([] if test_idx is None else list(test_idx)), dtype=int)
    )
    LOG.info(
        "[qrf_offline] start | scenario=%s | n_estimators=%d | min_samples_leaf=%d | cdf_n=%d | splits(train=%d calib=%d test=%d)",
        scenario_name,
        int(QRF_N_ESTIMATORS),
        int(QRF_MIN_SAMPLES_LEAF),
        int(QRF_CDF_N),
        len(train_idx),
        len(calib_idx),
        0 if test_idx is None else len(test_idx),
    )
    X_all, y_all, design_ids, w_all, stats_all = _build_episode_dataset(
        epman,
        grid,
        allowed_union,
        MAX_EPS_PER_DESIGN,
        MIN_EPS_PER_DESIGN,
        d_safe,
    )
    if y_all.size == 0:
        raise RuntimeError("No training rows found for QRF (check cache and MIN_EPS_PER_DESIGN).")

    train_mask = np.isin(design_ids, np.asarray(train_idx, dtype=int))
    calib_mask = np.isin(design_ids, np.asarray(calib_idx, dtype=int))
    test_mask = np.isin(design_ids, np.asarray(test_idx, dtype=int)) if test_idx is not None else np.zeros_like(train_mask, dtype=bool)

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    w_train = w_all[train_mask]
    X_calib = X_all[calib_mask]
    y_calib = y_all[calib_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]

    if y_train.size == 0:
        raise RuntimeError("No training rows found after split (check cache and split indices).")

    train_used = sorted(set(int(i) for i in design_ids[train_mask]))
    calib_used = sorted(set(int(i) for i in design_ids[calib_mask]))
    test_used = sorted(set(int(i) for i in design_ids[test_mask]))

    LOG.info(
        "[qrf_offline] dataset built in %.1fs | rows(train=%d calib=%d test=%d) | designs(train=%d calib=%d test=%d)",
        time.time() - t_start,
        int(y_train.size),
        int(y_calib.size),
        int(y_test.size),
        len(train_used),
        len(calib_used),
        len(test_used),
    )

    model = RandomForestQuantileRegressor(
        n_estimators=int(QRF_N_ESTIMATORS),
        min_samples_leaf=int(QRF_MIN_SAMPLES_LEAF),
        n_jobs=-1,
        random_state=int(seed),
    )
    t_fit = time.time()
    LOG.info("[qrf_offline] fitting QRF model...")
    model.fit(X_train, y_train, sample_weight=w_train)
    LOG.info("[qrf_offline] model fit done in %.1fs", time.time() - t_fit)

    t_cal = time.time()
    calib_deltas, calib_cov_pre, calib_cov_post = _calibrate_quantiles(
        model, X_calib, y_calib, CALIB_QUANTILES
    )
    LOG.info("[qrf_offline] quantile calibration done in %.1fs", time.time() - t_cal)
    wrapper: RandomForestQuantileRegressor | CalibratedQRF = model
    calib_source = None
    if calib_deltas:
        wrapper = CalibratedQRF(model, calib_deltas)
        calib_source = "calib"
    t_eval = time.time()
    calib_rmse, calib_mae = _eval_metrics(wrapper, X_calib, y_calib, quantile=EVAL_QUANTILE)
    test_rmse, test_mae = _eval_metrics(wrapper, X_test, y_test, quantile=EVAL_QUANTILE)
    eval_taus = np.linspace(0.001, 0.999, QRF_CDF_N, dtype=float)
    LOG.info("[qrf_offline] computing p-calibration and classification metrics on %d taus...", int(len(eval_taus)))
    p_calibration = _fit_p_calibrator(wrapper, X_all, y_all, design_ids, calib_mask, eval_taus)
    train_cls = _eval_classification_metrics(wrapper, X_train, y_train, eval_taus, p_calibration)
    calib_cls = _eval_classification_metrics(wrapper, X_calib, y_calib, eval_taus, p_calibration)
    test_cls = _eval_classification_metrics(wrapper, X_test, y_test, eval_taus, p_calibration)
    LOG.info("[qrf_offline] metrics computed in %.1fs", time.time() - t_eval)
    train_unsafe_rate = _unsafe_rate(y_train)
    calib_unsafe_rate = _unsafe_rate(y_calib)
    test_unsafe_rate = _unsafe_rate(y_test)
    LOG.info(
        "[qrf_offline] unsafe rate | train=%s calib=%s test=%s",
        train_unsafe_rate,
        calib_unsafe_rate,
        test_unsafe_rate,
    )
    LOG.info(
        "[qrf_offline] train metrics | logloss=%s acc=%s auc=%s",
        train_cls.get("logloss"),
        train_cls.get("accuracy"),
        train_cls.get("auc"),
    )
    LOG.info(
        "[qrf_offline] calib metrics | logloss=%s acc=%s auc=%s",
        calib_cls.get("logloss"),
        calib_cls.get("accuracy"),
        calib_cls.get("auc"),
    )
    LOG.info(
        "[qrf_offline] test metrics | logloss=%s acc=%s auc=%s",
        test_cls.get("logloss"),
        test_cls.get("accuracy"),
        test_cls.get("auc"),
    )

    if out_dir is None:
        out_dir = prepare_run_dir(surrogate_root, "qrf")
    else:
        os.makedirs(out_dir, exist_ok=True)
    if model_name is None:
        model_name = MODEL_OUT
    if not model_name.startswith(f"{scenario_name}_"):
        model_name = f"{scenario_name}_{model_name}"
    model_path = os.path.join(out_dir, model_name)
    dump(wrapper, model_path)

    meta = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "joblib_version": joblib.__version__,
        "scenario": scenario_name,
        "space_spec_name": space_spec.name,
        "space_spec_keys": list(space_spec.keys),
        "episode_root": str(ep_root),
        "surrogate_root": surrogate_root,
        "run_dir": out_dir,
        "dataset_manifest_path": str(Path(split_path).with_name("offline_artifacts.json")) if split_path else None,
        "split_path": split_path,
        "trainer_version": TRAINER_VERSION,
        "algo_version": algo_version,
        "model_id": Path(out_dir).name,
        "model_run_id": Path(out_dir).name,
        "model_family": Path(out_dir).parent.name,
        "model_path": model_path,
        "d_safe": d_safe,
        "min_eps_per_design": MIN_EPS_PER_DESIGN,
        "max_eps_per_design": MAX_EPS_PER_DESIGN,
        "n_estimators": int(QRF_N_ESTIMATORS),
        "min_samples_leaf": int(QRF_MIN_SAMPLES_LEAF),
        "eval_quantile": float(EVAL_QUANTILE),
        "calibration": {
            "source": calib_source,
            "quantiles": [float(q) for q in CALIB_QUANTILES],
            "deltas": {str(k): float(v) for k, v in calib_deltas.items()},
            "coverage_pre": {str(k): float(v) for k, v in calib_cov_pre.items()},
            "coverage_post": {str(k): float(v) for k, v in calib_cov_post.items()},
        },
        "train_idx": [int(x) for x in train_used],
        "calib_idx": [int(x) for x in calib_used],
        "test_idx": [int(x) for x in test_used],
        "row_counts": {
            "train": int(y_train.size),
            "calib": int(y_calib.size),
            "test": int(y_test.size),
        },
        "design_counts": {
            "train": int(len(train_used)),
            "calib": int(len(calib_used)),
            "test": int(len(test_used)),
        },
        "calib_rmse": calib_rmse,
        "calib_mae": calib_mae,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "unsafe_rate": {
            "train": train_unsafe_rate,
            "calib": calib_unsafe_rate,
            "test": test_unsafe_rate,
        },
        "classification_metrics": {
            "taus_n": int(QRF_CDF_N),
            "train": train_cls,
            "calib": calib_cls,
            "test": test_cls,
        },
        "cdf_n": int(QRF_CDF_N),
        "cdf_taus": [float(x) for x in eval_taus],
        "p_calibration": p_calibration,
        "stats": stats_all,
        "normalization": stats_all.get("norm"),
        "trace": {
            "model_id": Path(out_dir).name,
            "model_run_id": Path(out_dir).name,
            "model_family": Path(out_dir).parent.name,
            "dataset_manifest_path": str(Path(split_path).with_name("offline_artifacts.json")) if split_path else None,
            "split": split_trace(split_path) if split_path else None,
        },
    }
    meta_name = os.path.splitext(model_name)[0] + "_meta.json"
    meta_path = os.path.join(out_dir, meta_name)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    LOG.info("[qrf_offline] saved model=%s meta=%s | total_time=%.1fs", model_path, meta_path, time.time() - t_start)
    return {"model": model_path, "meta": meta_path}
