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
"""Coverage calibration shared by the offline Gaussian and Student-t GP trainers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import norm as stats_norm
from scipy.stats import t as stats_t


@dataclass
class MarginCalibrationCfg:
    """Defaults for global bias and variance-scale calibration on held-out margins."""

    calib_gamma: float = 0.9
    calib_max_iters: int = 40
    calib_tol: float = 1e-3
    calib_var_scale_bounds: Tuple[float, float] = (1e-3, 1e3)


def compute_gaussian_calibration(
    mu_all: np.ndarray,
    var_total_all: np.ndarray,
    margins: Dict[int, np.ndarray],
    calib_idx: np.ndarray,
    gamma: float,
    max_iters: int,
    tol: float,
    var_scale_bounds: Tuple[float, float],
    ) -> Tuple[float, float, int, int, float, float, float, np.ndarray]:
    """
    Fit global bias (median gbar - mu) and scalar variance scale for calibration.

    Only margin_coverage is supported: match target two-sided coverage on residuals.
    """
    if calib_idx.size == 0 or not margins:
        return 0.0, 1.0, 0, 0, float("nan"), float("nan"), float("nan"), np.array([], dtype=int)
    diffs: List[float] = []
    for gi in calib_idx:
        m = margins.get(int(gi))
        if m is None or len(m) == 0:
            continue
        diffs.append(float(np.mean(m)) - float(mu_all[int(gi)]))
    bias = float(np.median(diffs)) if diffs else 0.0
    eps = 1e-12
    used_designs: List[int] = []

    mu_list: List[float] = []
    sigma_list: List[float] = []
    g_list: List[float] = []
    for gi in calib_idx:
        m = margins.get(int(gi))
        if m is None or len(m) == 0:
            continue
        v_tot = float(var_total_all[int(gi)])
        if not (np.isfinite(v_tot) and v_tot > 0):
            continue
        mu_g = float(mu_all[int(gi)] + bias)
        sigma_g = float(np.sqrt(max(v_tot, eps)))
        used_designs.append(int(gi))
        for val in m:
            mu_list.append(mu_g)
            sigma_list.append(sigma_g)
            g_list.append(float(val))
    if len(g_list) == 0:
        return bias, 1.0, 0, 0, float("nan"), float("nan"), float("nan"), np.array([], dtype=int)

    mu_vec = np.asarray(mu_list, dtype=float)
    sigma_vec = np.asarray(sigma_list, dtype=float)
    g_vec = np.asarray(g_list, dtype=float)
    z = stats_norm.ppf(0.5 * (1.0 + gamma))

    def coverage(scale: float) -> float:
        s_safe = max(scale, eps)
        r = (g_vec - mu_vec) / (np.sqrt(s_safe) * sigma_vec + eps)
        return float(np.mean(np.abs(r) <= z))

    lo, hi = var_scale_bounds
    cov_lo = coverage(lo)
    cov_hi = coverage(hi)
    cov_lo_orig, cov_hi_orig = cov_lo, cov_hi
    best_scale = 1.0
    best_cov = coverage(best_scale)
    best_err = abs(best_cov - gamma)
    used_designs_unique = np.asarray(list(set(used_designs)), dtype=int)
    if cov_lo >= gamma:
        return bias, lo, len(used_designs_unique), len(g_vec), cov_lo_orig, cov_lo, cov_hi_orig, used_designs_unique
    if cov_hi <= gamma:
        return bias, hi, len(used_designs_unique), len(g_vec), cov_lo_orig, cov_hi, cov_hi_orig, used_designs_unique
    for _ in range(max_iters):
        mid = np.sqrt(lo * hi)
        cov_mid = coverage(mid)
        err_mid = abs(cov_mid - gamma)
        if err_mid < best_err:
            best_err = err_mid
            best_scale = mid
            best_cov = cov_mid
        if cov_mid < gamma:
            lo, cov_lo = mid, cov_mid
        else:
            hi, cov_hi = mid, cov_mid
        if best_err <= tol:
            break

    if not np.isfinite(best_scale) or best_scale <= 0:
        best_scale = 1.0
    return bias, best_scale, len(used_designs_unique), len(g_vec), cov_lo, best_cov, cov_hi, used_designs_unique


def compute_student_t_calibration(
    mu_all: np.ndarray,
    var_total_all: np.ndarray,
    margins: Dict[int, np.ndarray],
    calib_idx: np.ndarray,
    df: float,
    gamma: float,
    max_iters: int,
    tol: float,
    var_scale_bounds: Tuple[float, float],
    ) -> Tuple[float, float, int, int, float, float, float, np.ndarray]:
    """
    Fit global bias (median gbar - mu) and scalar variance scale for calibration.

    Only margin_coverage is supported: match target two-sided coverage on residuals.
    """
    if calib_idx.size == 0 or not margins:
        return 0.0, 1.0, 0, 0, float("nan"), float("nan"), float("nan"), np.array([], dtype=int)
    diffs: List[float] = []
    for gi in calib_idx:
        m = margins.get(int(gi))
        if m is None or len(m) == 0:
            continue
        diffs.append(float(np.mean(m)) - float(mu_all[int(gi)]))
    bias = float(np.median(diffs)) if diffs else 0.0
    eps = 1e-12
    used_designs: List[int] = []

    mu_list: List[float] = []
    sigma_list: List[float] = []
    g_list: List[float] = []
    for gi in calib_idx:
        m = margins.get(int(gi))
        if m is None or len(m) == 0:
            continue
        v_tot = float(var_total_all[int(gi)])
        if not (np.isfinite(v_tot) and v_tot > 0):
            continue
        mu_g = float(mu_all[int(gi)] + bias)
        sigma_g = float(np.sqrt(max(v_tot, eps)))
        used_designs.append(int(gi))
        for val in m:
            mu_list.append(mu_g)
            sigma_list.append(sigma_g)
            g_list.append(float(val))
    if len(g_list) == 0:
        return bias, 1.0, 0, 0, float("nan"), float("nan"), float("nan"), np.array([], dtype=int)

    mu_vec = np.asarray(mu_list, dtype=float)
    sigma_vec = np.asarray(sigma_list, dtype=float)
    g_vec = np.asarray(g_list, dtype=float)
    df = float(df)
    if not (df > 2.0):
        raise ValueError(f"Student-t df must be > 2 for finite variance, got df={df}")
    t_factor = math.sqrt((df - 2.0) / df)
    z = stats_t.ppf(0.5 * (1.0 + gamma), df) * t_factor

    def coverage(scale: float) -> float:
        s_safe = max(scale, eps)
        r = (g_vec - mu_vec) / (np.sqrt(s_safe) * sigma_vec + eps)
        return float(np.mean(np.abs(r) <= z))

    lo, hi = var_scale_bounds
    cov_lo = coverage(lo)
    cov_hi = coverage(hi)
    cov_lo_orig, cov_hi_orig = cov_lo, cov_hi
    best_scale = 1.0
    best_cov = coverage(best_scale)
    best_err = abs(best_cov - gamma)
    used_designs_unique = np.asarray(list(set(used_designs)), dtype=int)
    if cov_lo >= gamma:
        return bias, lo, len(used_designs_unique), len(g_vec), cov_lo_orig, cov_lo, cov_hi_orig, used_designs_unique
    if cov_hi <= gamma:
        return bias, hi, len(used_designs_unique), len(g_vec), cov_lo_orig, cov_hi, cov_hi_orig, used_designs_unique
    for _ in range(max_iters):
        mid = np.sqrt(lo * hi)
        cov_mid = coverage(mid)
        err_mid = abs(cov_mid - gamma)
        if err_mid < best_err:
            best_err = err_mid
            best_scale = mid
            best_cov = cov_mid
        if cov_mid < gamma:
            lo, cov_lo = mid, cov_mid
        else:
            hi, cov_hi = mid, cov_mid
        if best_err <= tol:
            break

    if not np.isfinite(best_scale) or best_scale <= 0:
        best_scale = 1.0
    return bias, best_scale, len(used_designs_unique), len(g_vec), cov_lo, best_cov, cov_hi, used_designs_unique
