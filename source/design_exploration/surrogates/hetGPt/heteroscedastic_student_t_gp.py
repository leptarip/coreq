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
Heteroscedastic Student-t surrogate for continuous safety margin:
    m(design, episode) = min_d - d_safe

This module is a pragmatic heavy-tailed variant of `heteroscedastic_gp.py`.

Key idea:
- We keep the same mean GP and variance GP machinery (sklearn GaussianProcessRegressor).
- We interpret the predictive distribution of the margin as Student-t with a
  FIXED degrees-of-freedom `df`, matching the predicted mean and total variance.
- Violation probability is computed via the Student-t CDF: P(m < 0).

Why fixed df?
- Per-design MLE fits of df/scale can be numerically unstable with ~30-60 replicas.
- A fixed heavy-tailed df (e.g., 4–8) provides a robust baseline with minimal knobs.

Important:
This surrogate is for *ranking / proposing candidate safe sets*. Your holdout audit
provides the distribution-free guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import t as stats_t

from source.design_exploration.commons import design_space
from source.design_exploration.commons.episodes.episode_manager import EpisodeManager
from source.design_exploration.commons.episodes.episode_result import EpisodeResult


@dataclass
class HeteroStudentTCfg:
    # Core
    seed: int = 0
    d_safe: float = 0.5

    # Student-t degrees of freedom (must be > 2 to have finite variance)
    df: float = 5.0

    # GP hyperparameters / numerics (copied from HeteroGPCfg)
    noise_floor: float = 1e-5          # min alpha passed to the GP
    var_floor: float = 1e-6            # min episode variance before log
    length_scale: float = 1.0
    length_scale_bounds: Tuple[float, float] = (1e-6, 1e4)
    nu: float = 2.5                    # smoothness for Matern kernel
    const_kernel: float = 1.0
    n_restarts: int = 3
    normalize_X: bool = True


@dataclass
class MarginDataset:
    X: np.ndarray                    # possibly normalized features
    X_raw: np.ndarray                # unnormalized features (for reference)
    y: np.ndarray                    # mean margin per design
    alpha_mean: np.ndarray           # variance of the mean (heteroscedastic alpha = var_episode / n)
    reps_count: np.ndarray           # number of replicas per design
    var_episode: np.ndarray          # sample variance of single-episode margins per design (aleatoric)
    log_var_episode: np.ndarray      # log(var_episode + var_floor) for variance GP
    gidx: np.ndarray                 # design indices
    fallback_var_mask: np.ndarray    # True if variance fell back (e.g., <2 reps)
    norm_lo: Optional[np.ndarray]    # normalization shift (if applied)
    norm_span: Optional[np.ndarray]  # normalization scale (if applied)
    margins: Dict[int, np.ndarray]   # raw per-episode margins per design


def _payload_to_episode_result(payload_dict) -> EpisodeResult:
    ep = EpisodeResult()
    ep.result = payload_dict if isinstance(payload_dict, dict) else {"payload": payload_dict}
    return ep


def _margin_from_episode(ep: EpisodeResult, d_safe: float) -> Optional[float]:
    """Extract m = min_d - d_safe; return None if the metric is missing."""
    safety = getattr(ep, "result", {}) or {}
    safety = safety.get("safety_m", {}) or {}
    try:
        min_d = float(safety.get("min_d"))
    except (TypeError, ValueError):
        return None
    return float(min_d) - float(d_safe)


def build_margin_dataset(
    epman: EpisodeManager,
    cfg: HeteroStudentTCfg,
    min_episodes_per_design: int = 1,
    norm_grid: Optional[np.ndarray] = None,
    allowed_idx: Optional[Sequence[int]] = None,
) -> MarginDataset:
    """
    Aggregate cached episodes into a training set of margins.

    - Uses all designs with at least `min_episodes_per_design` margin observations.
    - y is the sample mean of margins; alpha_mean is the variance of that mean
      (sample variance / n) with a small floor to keep the GP well-conditioned.
    """
    grid = np.asarray(epman.GRID, dtype=float)
    counts = epman.list_gidx_with_counts()
    allowed_set = set(int(i) for i in allowed_idx) if allowed_idx is not None else None

    X_raw: List[np.ndarray] = []
    y: List[float] = []
    gidx: List[int] = []
    n_eps: List[int] = []
    margins_by_gidx: Dict[int, np.ndarray] = {}
    var_eps: List[float] = []
    fallback_mask: List[bool] = []

    for gi in sorted(counts.keys()):
        if allowed_set is not None and int(gi) not in allowed_set:
            continue
        cached = epman.episodes(gi, consume=False)
        margins: List[float] = []
        for row in cached:
            payload = row.get("payload", {})
            margin_val = _margin_from_episode(_payload_to_episode_result(payload), cfg.d_safe)
            if margin_val is not None:
                margins.append(float(margin_val))

        if len(margins) < min_episodes_per_design:
            continue

        margins_arr = np.asarray(margins, dtype=float)
        mean_g = float(np.mean(margins_arr))
        has_var = len(margins_arr) > 1
        var_g = float(np.var(margins_arr, ddof=1)) if has_var else float("nan")

        X_raw.append(grid[int(gi)])
        y.append(mean_g)
        var_eps.append(var_g)
        fallback_mask.append(not has_var)
        gidx.append(int(gi))
        n_eps.append(len(margins_arr))
        margins_by_gidx[int(gi)] = margins_arr

    if not X_raw:
        raise ValueError("No usable margin data found in the episode cache.")

    # Fallback variance for designs with <2 reps: median of valid variances, else small constant.
    var_array = np.asarray(var_eps, dtype=float)
    valid = ~np.isnan(var_array)
    fallback = float(np.median(var_array[valid])) if np.any(valid) else 1e-4
    var_array = np.where(valid, var_array, fallback)
    var_array = np.maximum(var_array, cfg.var_floor)

    n_eps_arr = np.asarray(n_eps, dtype=int)
    alpha_mean_arr = np.maximum(var_array / np.maximum(n_eps_arr, 1), cfg.noise_floor)
    log_var_arr = np.log(var_array + cfg.var_floor)
    fallback_arr = np.asarray(fallback_mask, dtype=bool)

    X_raw_arr = np.vstack(X_raw).astype(float)

    # Use a fixed normalization based on the full grid (or provided norm_grid) to avoid drift across AL iters.
    lo = span = None
    if cfg.normalize_X:
        full = np.asarray(norm_grid if norm_grid is not None else epman.GRID, dtype=float)
        _, lo, span = design_space.normalize_grid(full)
        eps = 1e-12
        span_safe = np.where(span == 0.0, eps, span)
        X_norm = (X_raw_arr - lo) / span_safe
        assert np.isfinite(X_norm).all(), "Non-finite normalized inputs detected; check normalization span."
    else:
        X_norm = X_raw_arr
        span_safe = span

    return MarginDataset(
        X=X_norm,
        X_raw=X_raw_arr,
        y=np.asarray(y, dtype=float),
        alpha_mean=alpha_mean_arr,
        reps_count=n_eps_arr,
        var_episode=var_array,
        log_var_episode=log_var_arr,
        gidx=np.asarray(gidx, dtype=int),
        fallback_var_mask=fallback_arr,
        norm_lo=lo,
        norm_span=span_safe,
        margins=margins_by_gidx,
    )


def fit_heteroscedastic_gp(dataset: MarginDataset, cfg: HeteroStudentTCfg) -> GaussianProcessRegressor:
    """Fit a GP with per-point noise (alpha_mean) to the provided margin dataset."""
    kernel = (
        ConstantKernel(cfg.const_kernel, (1e-6, 1e5))
        * Matern(length_scale=cfg.length_scale, length_scale_bounds=cfg.length_scale_bounds, nu=cfg.nu)
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=np.asarray(dataset.alpha_mean, dtype=float),
        normalize_y=True,
        random_state=cfg.seed,
        n_restarts_optimizer=cfg.n_restarts,
    )
    model.fit(dataset.X, dataset.y)
    return model


def fit_variance_gp(dataset: MarginDataset, cfg: HeteroStudentTCfg) -> GaussianProcessRegressor:
    """
    Fit a GP to predict log(var_episode) (aleatoric component).
    Observation noise scales down with replica count to reflect uncertainty of
    the variance estimate: alpha_i ~ 1/(n_i - 1), floored by cfg.noise_floor.
    """
    kernel = (
        ConstantKernel(cfg.const_kernel, (1e-6, 1e5))
        * Matern(length_scale=cfg.length_scale, length_scale_bounds=cfg.length_scale_bounds, nu=cfg.nu)
    )
    reps = np.maximum(dataset.reps_count.astype(float), 1.0)
    alpha_var = np.maximum(1.0 / np.maximum(reps - 1.0, 1.0), cfg.noise_floor)
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha_var,
        normalize_y=True,
        random_state=cfg.seed,
        n_restarts_optimizer=cfg.n_restarts,
    )
    model.fit(dataset.X, dataset.log_var_episode)
    return model


def predict_margin(
    model: GaussianProcessRegressor,
    GRID: np.ndarray,
    norm_params: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    return_std: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict margin over a design grid.

    Args:
        model: trained GP.
        GRID: full design grid (raw scale).
        norm_params: optional (lo, span) tuple from MarginDataset if normalization was used.
        return_std: if True, also return predictive stddev.
    """
    X = np.asarray(GRID, dtype=float)
    if norm_params is not None and norm_params[0] is not None:
        lo, span = norm_params
        eps = 1e-12
        span_safe = np.where(span == 0.0, eps, span)
        X = (X - lo) / span_safe
    if return_std:
        mean, std = model.predict(X, return_std=True)
        return mean, std
    mean = model.predict(X, return_std=False)
    return mean, None


def _t_scale_from_variance(var: np.ndarray, df: float, eps: float = 1e-12) -> np.ndarray:
    """
    Convert desired variance into Student-t 'scale' parameter such that
        Var[t_df(loc, scale)] = var.

    For df>2: Var = scale^2 * df/(df-2)  =>  scale = sqrt(var * (df-2)/df).
    """
    df = float(df)
    if not (df > 2.0):
        raise ValueError(f"Student-t df must be > 2 for finite variance, got df={df}")
    var = np.asarray(var, dtype=float)
    var = np.maximum(var, 0.0)
    return np.sqrt(var * (df - 2.0) / df + eps)


def predict_violation_probability(
    mean_model: GaussianProcessRegressor,
    var_model: GaussianProcessRegressor,
    GRID: np.ndarray,
    norm_params: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    df: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict mean margin, epistemic stddev, aleatoric variance, and violation probability
    under a fixed-df Student-t predictive distribution.

    Returns:
        mu: predicted mean margin
        std_mean: epistemic stddev from the mean GP
        var_ep: predicted per-episode (aleatoric) variance from variance GP
        prob_violate: P(margin < 0)
    """
    mu, std_mean = predict_margin(mean_model, GRID, norm_params=norm_params, return_std=True)

    X = np.asarray(GRID, dtype=float)
    if norm_params is not None:
        lo, span = norm_params
        eps = 1e-12
        span_safe = np.where(span == 0.0, eps, span)
        X = (X - lo) / span_safe

    log_var_ep = var_model.predict(X, return_std=False)
    var_ep = np.exp(log_var_ep)

    var_mean = np.square(std_mean)
    var_total = var_mean + var_ep

    df_use = float(df) if df is not None else 5.0
    scale_total = _t_scale_from_variance(var_total, df=df_use)

    # P(m < 0) where m ~ t_df(loc=mu, scale=scale_total)
    prob_violate = stats_t.cdf(0.0, df=df_use, loc=mu, scale=scale_total)
    return mu, std_mean, var_ep, prob_violate


def train_from_episode_cache(
    epman: EpisodeManager,
    cfg: Optional[HeteroStudentTCfg] = None,
    min_episodes_per_design: int = 1,
    allowed_idx: Optional[Sequence[int]] = None,
) -> Tuple[GaussianProcessRegressor, GaussianProcessRegressor, MarginDataset]:
    """
    Convenience wrapper: build the margin dataset from cached episodes and fit the GP pair.
    """
    cfg = cfg or HeteroStudentTCfg()
    dataset = build_margin_dataset(
        epman,
        cfg=cfg,
        min_episodes_per_design=min_episodes_per_design,
        norm_grid=epman.GRID,
        allowed_idx=allowed_idx,
    )
    mean_model = fit_heteroscedastic_gp(dataset, cfg=cfg)
    var_model = fit_variance_gp(dataset, cfg=cfg)
    return mean_model, var_model, dataset


# -----------------------------------------------------------------------------
# sklearn-like wrapper
# -----------------------------------------------------------------------------


class HeteroStudentTGPClassifier:
    """Sklearn-like wrapper exposing ``predict_proba`` for :mod:`audit.py`.

    The wrapped surrogate predicts a *continuous safety margin* m(x) and its
    uncertainty. We interpret the predictive distribution of the margin as a
    fixed-df Student-t distribution:

        margin(x) ~ StudentT(df, loc=mu(x), scale=s(x))

    where the scale is chosen so that Var[margin(x)] matches the total
    predictive variance (epistemic + aleatoric).

    ``predict_proba`` returns two columns:
        - proba[:, 0] = P(unsafe) = P(margin < 0)
        - proba[:, 1] = P(safe)   = 1 - P(unsafe)

    This matches the expectation in ``audit.py`` that
    ``model.predict_proba(X)[:, 1]`` is a monotone *safety score*.

    Notes
    -----
    * ``bias`` and ``var_scale`` are optional global calibration parameters.
      If you didn't calibrate, leave them at (0, 1).
    * The audit guarantee remains distribution-free; this wrapper only produces
      scores used to propose candidate sets.
    """

    def __init__(
        self,
        mean_model: GaussianProcessRegressor,
        var_model: GaussianProcessRegressor,
        *,
        norm_lo: Optional[np.ndarray] = None,
        norm_span: Optional[np.ndarray] = None,
        df: float = 5.0,
        bias: float = 0.0,
        var_scale: float = 1.0,
        clip: bool = True,
    ) -> None:
        self.mean_model = mean_model
        self.var_model = var_model
        self.norm_lo = None if norm_lo is None else np.asarray(norm_lo, dtype=float)
        self.norm_span = None if norm_span is None else np.asarray(norm_span, dtype=float)
        self.df = float(df)
        self.bias = float(bias)
        self.var_scale = float(var_scale)
        self.clip = bool(clip)

        # sklearn convention
        self.classes_ = np.asarray([0, 1], dtype=int)

    @classmethod
    def from_dataset(
        cls,
        mean_model: GaussianProcessRegressor,
        var_model: GaussianProcessRegressor,
        dataset: MarginDataset,
        *,
        df: Optional[float] = None,
        bias: float = 0.0,
        var_scale: float = 1.0,
        clip: bool = True,
    ) -> "HeteroStudentTGPClassifier":
        """Construct a wrapper from a trained (mean,var) GP pair and dataset."""
        df_use = float(df) if df is not None else 5.0
        return cls(
            mean_model,
            var_model,
            norm_lo=getattr(dataset, "norm_lo", None),
            norm_span=getattr(dataset, "norm_span", None),
            df=df_use,
            bias=bias,
            var_scale=var_scale,
            clip=clip,
        )

    def _as_2d(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must be 1D or 2D array-like, got shape={X.shape}")
        return X

    def _normalize(self, X_raw: np.ndarray) -> np.ndarray:
        if self.norm_lo is None or self.norm_span is None:
            return X_raw
        eps = 1e-12
        span_safe = np.where(self.norm_span == 0.0, eps, self.norm_span)
        return (X_raw - self.norm_lo) / span_safe

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_raw = self._as_2d(X)
        Xn = self._normalize(X_raw)

        mu, std_mean = self.mean_model.predict(Xn, return_std=True)
        mu = np.asarray(mu, dtype=float) + self.bias
        std_mean = np.asarray(std_mean, dtype=float)

        log_var_ep = self.var_model.predict(Xn, return_std=False)
        var_ep = np.exp(np.asarray(log_var_ep, dtype=float))

        var_total = (std_mean ** 2) + var_ep
        var_total = np.maximum(var_total, 1e-12) * max(self.var_scale, 1e-12)

        scale_total = _t_scale_from_variance(var_total, df=self.df)
        p_unsafe = stats_t.cdf(0.0, df=self.df, loc=mu, scale=scale_total)
        p_safe = 1.0 - p_unsafe

        if self.clip:
            p_unsafe = np.clip(p_unsafe, 0.0, 1.0)
            p_safe = np.clip(p_safe, 0.0, 1.0)

        return np.stack([p_unsafe, p_safe], axis=1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return a 1D safety score (higher is safer)."""
        return self.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction based on ``P(safe) >= threshold``."""
        return (self.predict_proba(X)[:, 1] >= float(threshold)).astype(int)


