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
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from source.design_exploration.commons.episodes.episode_result import EpisodeResult


DEFAULT_RHO = 0.05


@dataclass(frozen=True)
class ParEGOProfileSpec:
    """The optimizer policy, recorded verbatim in every run manifest."""

    profile: str
    acquisition: str
    n_initial_points: int
    surrogate_target: str
    cost_normalization: str
    performance_normalization: str
    tie_break: str
    degenerate_prediction_policy: str


# Scalarized lower-quantile ParEGO: ten random initial designs, then the QRF is
# fitted to the augmented Chebyshev scalar of min-max normalized objectives and
# the candidate with the lowest predicted lower quantile is evaluated. Exact ties
# are broken uniformly with the run-seeded RNG.
PROFILE = "scalarized_leaf3_seeded"
PROFILE_SPEC = ParEGOProfileSpec(
    profile=PROFILE,
    acquisition="lower_quantile",
    n_initial_points=10,
    surrogate_target="scalarized_objectives",
    cost_normalization="observed_minmax",
    performance_normalization="observed_minmax",
    tie_break="seeded_random",
    degenerate_prediction_policy="seeded_random",
)


def resolve_parego_profile(profile: Optional[str] = None) -> ParEGOProfileSpec:
    """Return the optimizer policy; ``profile`` may only name it."""
    if profile is not None and str(profile) != PROFILE:
        raise ValueError(f"unknown ParEGO profile {profile!r}; the only profile is {PROFILE!r}")
    return PROFILE_SPEC


def _payload_to_episode_result(payload_dict: dict) -> EpisodeResult:
    ep = EpisodeResult()
    ep.result = payload_dict if isinstance(payload_dict, dict) else {"payload": payload_dict}
    return ep


def _extract_metric(ep: EpisodeResult, metric_path: str) -> Optional[float]:
    cur: Any = ep.result
    for part in metric_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    try:
        val = float(cur)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return val


def _clean_values(values: Sequence[Optional[float]]) -> np.ndarray:
    vals = [v for v in values if v is not None]
    if not vals:
        return np.asarray([], dtype=float)
    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def _aggregate(values: Sequence[Optional[float]], mode: str) -> float:
    vals = _clean_values(values)
    if vals.size == 0:
        return float("nan")
    if mode == "max":
        return float(np.max(vals))
    if mode == "mean":
        return float(np.mean(vals))
    if mode == "median":
        return float(np.median(vals))
    return float(np.min(vals))


def _aggregate_with_noise(values: Sequence[Optional[float]], mode: str) -> Tuple[float, Optional[float]]:
    vals = _clean_values(values)
    if vals.size == 0:
        return float("nan"), None
    if mode == "mean":
        mean = float(np.mean(vals))
        if len(vals) > 1:
            var = float(np.var(vals, ddof=1) / len(vals))
        else:
            var = 0.0
        return mean, var
    if mode == "max":
        return float(np.max(vals)), None
    if mode == "median":
        return float(np.median(vals)), None
    return float(np.min(vals)), None


def _portfolio_noise(
    scenario_variances: Sequence[Optional[float]], mode: str
) -> Optional[float]:
    """Variance of an equally weighted scenario mean, when it is identifiable."""
    if mode != "mean" or not scenario_variances or any(
        value is None or not np.isfinite(value) for value in scenario_variances
    ):
        return None
    count = len(scenario_variances)
    return float(np.sum(np.asarray(scenario_variances, dtype=float)) / (count * count))


def _non_dominated_mask(costs: np.ndarray, perfs: np.ndarray) -> np.ndarray:
    n = len(costs)
    mask = np.ones(n, dtype=bool)
    for j in range(n):
        if not mask[j]:
            continue
        for k in range(n):
            if j == k or not mask[k]:
                continue
            if costs[k] <= costs[j] and perfs[k] <= perfs[j] and (
                costs[k] < costs[j] or perfs[k] < perfs[j]
            ):
                mask[j] = False
                break
    return mask


def scalarize(Y: np.ndarray, weights: np.ndarray, rho: float = DEFAULT_RHO) -> np.ndarray:
    """
    Collapse multi-objective observations to the augmented Chebyshev scalar.

    Objectives are rescaled to [0, 1] over their observed range first, as ParEGO
    requires; otherwise the objective with the wider raw range decides the
    Chebyshev term by itself.
    """
    Y = np.asarray(Y, dtype=float)
    if not np.isfinite(float(rho)) or float(rho) < 0.0:
        raise ValueError("rho must be finite and non-negative")
    lo = np.min(Y, axis=0)
    span = np.max(Y, axis=0) - lo
    Y_norm = np.where(span > 0, (Y - lo) / np.where(span > 0, span, 1.0), 0.0)
    weighted = weights * Y_norm
    return np.max(weighted, axis=1) + rho * np.sum(weighted, axis=1)


def sample_weights(rng: np.random.Generator, num_objectives: int) -> np.ndarray:
    """
    Draw scalarization weights uniformly from the simplex.

    Normalizing independent uniforms does not do that -- for two objectives it
    peaks at (0.5, 0.5) -- which under-samples the ends of the front.
    """
    return rng.dirichlet(np.ones(num_objectives))


def _finite_argmin(values: np.ndarray, label: str, *, rng: np.random.Generator) -> Tuple[int, int, bool]:
    """Index of the smallest finite value; exact ties are broken with ``rng``.

    Returns ``(index, tied_minimum_count, degenerate_surface)``.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise RuntimeError(f"QRF returned no finite {label} candidate predictions")
    extreme = float(np.min(values[finite]))
    tied = np.flatnonzero(finite & np.isclose(values, extreme, rtol=1e-12, atol=1e-12))
    selected = int(rng.choice(tied))
    return selected, int(len(tied)), bool(len(tied) == int(finite.sum()))


class ParEGOQuantileRF:
    def __init__(
        self,
        X_candidates: np.ndarray,
        candidate_indices: np.ndarray,
        num_objectives: int = 2,
        rho: float = DEFAULT_RHO,
        random_seed: Optional[int] = None,
        n_estimators: int = 200,
        min_samples_leaf: int = 3,
        quantile: float = 0.25,
        profile: Optional[str] = None,
    ):
        profile_spec = resolve_parego_profile(profile)
        assert 0.0 < quantile < 0.5, (
            "quantile should be in (0, 0.5) for lower-quantile minimization"
        )
        if not np.isfinite(float(rho)) or float(rho) < 0.0:
            raise ValueError("rho must be finite and non-negative")

        self.X_map = X_candidates
        self.indices_map = candidate_indices
        self.num_objs = num_objectives
        self.rho = rho
        self.quantile = quantile
        self.profile = profile_spec.profile
        self.acquisition = profile_spec.acquisition
        self.tie_break = profile_spec.tie_break
        self.degenerate_prediction_policy = profile_spec.degenerate_prediction_policy
        self.n_initial_points = profile_spec.n_initial_points
        self.rng = np.random.default_rng(random_seed)

        self.observed_indices: List[int] = []
        self.observed_y: List[np.ndarray] = []
        self.observed_noise: List[Optional[float]] = []

        self.gidx_to_row = {gidx: i for i, gidx in enumerate(self.indices_map)}
        self.done = False
        self.last_acq: Optional[float] = None
        self.last_weights: Optional[np.ndarray] = None
        self._clear_acquisition_diagnostics()

        try:
            from quantile_forest import RandomForestQuantileRegressor
        except Exception as exc:
            raise RuntimeError(
                "quantile_forest is required to run ParEGO QRF optimization."
            ) from exc

        self.rf = RandomForestQuantileRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=random_seed,
        )

    def _clear_acquisition_diagnostics(self) -> None:
        self.last_surrogate_quantile_span: Optional[float] = None
        self.last_surrogate_prediction_degenerate: Optional[bool] = None
        self.last_acquisition_score_span: Optional[float] = None
        self.last_acquisition_score_degenerate: Optional[bool] = None
        self.last_tie_candidate_count: Optional[int] = None
        self.last_tie_break_policy: Optional[str] = None
        self.last_selection_reason: Optional[str] = None
        self.last_forest_split_tree_fraction: Optional[float] = None
        self.last_forest_mean_depth: Optional[float] = None
        self.last_forest_max_depth: Optional[int] = None

    def _record_selection_diagnostics(
        self, q_low: np.ndarray, tie_count: int, degenerate: bool
    ) -> None:
        finite = np.asarray(q_low, dtype=float)
        finite = finite[np.isfinite(finite)]
        self.last_surrogate_quantile_span = float(np.ptp(finite))
        self.last_surrogate_prediction_degenerate = bool(
            np.isclose(self.last_surrogate_quantile_span, 0.0, rtol=0.0, atol=1e-12)
        )
        self.last_acquisition_score_span = self.last_surrogate_quantile_span
        self.last_acquisition_score_degenerate = bool(degenerate)
        self.last_tie_candidate_count = int(tie_count)
        self.last_tie_break_policy = self.tie_break
        if degenerate:
            prefix = "degenerate_surface"
        elif tie_count > 1:
            prefix = "tied_extreme"
        else:
            prefix = "unique_extreme"
        self.last_selection_reason = "%s_%s" % (prefix, self.tie_break)

    def tell(self, global_gidx: int, y_vals: List[float], noise_variance: Optional[float] = None):
        values = np.asarray(y_vals, dtype=float).reshape(-1)
        if values.shape != (self.num_objs,) or not np.all(np.isfinite(values)):
            raise ValueError("tell objectives must be a finite vector of the declared width")
        self.observed_indices.append(global_gidx)
        self.observed_y.append(values)
        self.observed_noise.append(noise_variance)

    def ask(self) -> Optional[int]:
        if self.done:
            return None
        if not self.observed_indices:
            self.last_acq = None
            self.last_weights = None
            self._clear_acquisition_diagnostics()
            return int(self.rng.choice(self.indices_map))

        Y = np.array(self.observed_y)
        weights = sample_weights(self.rng, self.num_objs)
        self.last_weights = weights.copy()
        fitted_targets = scalarize(Y, weights, rho=self.rho)

        X_train = []
        y_train = []
        for gidx, target in zip(self.observed_indices, fitted_targets):
            if gidx in self.gidx_to_row:
                X_train.append(self.X_map[self.gidx_to_row[gidx]])
                y_train.append(target)
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        mask_unobs = np.isin(self.indices_map, self.observed_indices, invert=True)
        if not np.any(mask_unobs):
            self.done = True
            return None

        if X_train.shape[0] < self.n_initial_points:
            self.last_acq = None
            self._clear_acquisition_diagnostics()
            return int(self.rng.choice(self.indices_map[mask_unobs]))

        self.rf.fit(X_train, y_train)
        tree_depths = np.asarray(
            [tree.tree_.max_depth for tree in self.rf.estimators_], dtype=float
        )
        self.last_forest_split_tree_fraction = float(np.mean(tree_depths > 0.0))
        self.last_forest_mean_depth = float(np.mean(tree_depths))
        self.last_forest_max_depth = int(np.max(tree_depths))

        X_test = self.X_map[mask_unobs]
        cand_indices = self.indices_map[mask_unobs]
        q_low = np.asarray(
            self.rf.predict(X_test, quantiles=self.quantile), dtype=float
        ).reshape(-1)
        best_local, tie_count, degenerate = _finite_argmin(q_low, "lower-quantile", rng=self.rng)
        self._record_selection_diagnostics(q_low, tie_count, degenerate)
        self.last_acq = float(-q_low[best_local])
        return int(cand_indices[best_local])
