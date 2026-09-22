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
Predict QRF p(unsafe) from cached offline model.

Intended to run under Python 3.11+ with quantile_forest installed.
Used by eval scripts when the main environment cannot import QRF classes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from quantile_forest import RandomForestQuantileRegressor  # noqa: F401

# Ensure repository root on sys.path so joblib can resolve CalibratedQRF
_HERE = Path(__file__).resolve()
for cand in [_HERE] + list(_HERE.parents):
    if (cand / "source").is_dir() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
        break

from source.design_exploration.surrogates.qrf import train_qrf_offline  # noqa: F401


def _register_calibrated_qrf_aliases() -> None:
    """
    Ensure CalibratedQRF is resolvable under common module paths used by pickles.
    When train_qrf_offline.py is run as a script, the class is pickled as __main__.CalibratedQRF.
    """
    import types

    cls = train_qrf_offline.CalibratedQRF
    for mod_name in (
        "__main__",
    ):
        mod = sys.modules.get(mod_name)
        if mod is None:
            mod = types.ModuleType(mod_name)
            sys.modules[mod_name] = mod
        if not hasattr(mod, "CalibratedQRF"):
            setattr(mod, "CalibratedQRF", cls)


def _qrf_norm_params(norm_dict: Optional[dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not norm_dict:
        return None, None
    g_min = norm_dict.get("g_min")
    g_range = norm_dict.get("g_range")
    g_max = norm_dict.get("g_max")
    if g_min is None:
        return None, None
    g_min_arr = np.asarray(g_min, dtype=float)
    if g_range is None and g_max is not None:
        g_range = np.asarray(g_max, dtype=float) - g_min_arr
    if g_range is None:
        return g_min_arr, None
    g_range_arr = np.asarray(g_range, dtype=float)
    g_range_arr = np.where(g_range_arr > 0, g_range_arr, 1.0)
    return g_min_arr, g_range_arr


def _apply_qrf_norm(X: np.ndarray, g_min: Optional[np.ndarray], g_range: Optional[np.ndarray]) -> np.ndarray:
    if g_min is None or g_range is None:
        return X
    return (X - g_min) / g_range


def _qrf_predict_p_unsafe(model, X: np.ndarray, taus: np.ndarray) -> np.ndarray:
    qs = np.asarray(taus, dtype=float).reshape(-1)
    qs = np.sort(np.unique(qs))
    preds = np.asarray(model.predict(X, quantiles=qs.tolist()), dtype=float)
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


def _apply_p_calibration(p: np.ndarray, calib: Optional[dict]) -> np.ndarray:
    if not calib:
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


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Predict QRF p(unsafe) under Python 3.11.")
    ap.add_argument("--model", required=True, help="Path to QRF joblib model.")
    ap.add_argument("--meta", default=None, help="Optional QRF meta JSON for normalization.")
    ap.add_argument("--x-path", required=True, help="Input X .npy path")
    ap.add_argument("--taus-path", default=None, help="Optional quantile grid .npy")
    ap.add_argument("--out-path", required=True, help="Output .npy path")
    ap.add_argument("--cdf-n", type=int, default=999, help="Quantile grid size if --taus-path not set")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    _register_calibrated_qrf_aliases()
    X = np.load(args.x_path)
    taus = None
    if args.taus_path:
        taus = np.load(args.taus_path)
    if taus is None:
        taus = np.linspace(0.001, 0.999, max(2, int(args.cdf_n)), dtype=float)

    meta = {}
    if args.meta:
        try:
            with open(args.meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except FileNotFoundError:
            meta = {}
    norm = meta.get("normalization") or meta.get("stats", {}).get("norm")
    p_calib = meta.get("p_calibration")
    g_min, g_range = _qrf_norm_params(norm)
    Xn = _apply_qrf_norm(np.asarray(X, dtype=float), g_min, g_range)

    model = joblib.load(args.model)
    p = _qrf_predict_p_unsafe(model, Xn, taus)
    p = _apply_p_calibration(p, p_calib)
    np.save(args.out_path, p)


if __name__ == "__main__":
    main()
