# Copyright (c) 2026 278097159+leptarip@users.noreply.github.com
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

"""The ParEGO scalarization, weight sampling and tie-breaking."""
from __future__ import annotations

import numpy as np
import pytest

from source.design_exploration.optimization.parego_qrf_common import (
    ParEGOQuantileRF,
    _portfolio_noise,
    resolve_parego_profile,
    sample_weights,
    scalarize,
)

# Cost and velocity on the scales the optimizer actually sees: cost spans
# thousands, velocity spans about 1.6.
OBSERVED = np.array([(609.0, -1.20), (700.0, -1.35), (873.0, -1.62), (650.0, -1.30)])
EVEN = np.array([0.5, 0.5])


def test_scalarization_is_invariant_to_objective_units():
    rescaled = OBSERVED * np.array([1.0, 1000.0])
    assert np.array_equal(
        np.argsort(scalarize(OBSERVED, EVEN)), np.argsort(scalarize(rescaled, EVEN))
    )


def test_the_wider_objective_does_not_decide_on_its_own():
    lo = OBSERVED.min(axis=0)
    z_star = lo - 0.05 * (OBSERVED.max(axis=0) - lo)
    raw = EVEN * (OBSERVED - z_star)
    assert np.all(raw[:, 0] > raw[:, 1]), "precondition: unscaled, cost dominates every row"

    span = OBSERVED.max(axis=0) - lo
    normalized = EVEN * ((OBSERVED - lo) / span)
    assert not np.all(normalized[:, 0] >= normalized[:, 1])


def test_weights_are_uniform_on_the_simplex():
    rng = np.random.default_rng(0)
    weights = np.array([sample_weights(rng, 2)[0] for _ in range(20000)])
    # A uniform marginal puts a tenth of its mass in each decile.
    assert abs(np.mean(weights < 0.1) - 0.1) < 0.02


def test_degenerate_range_does_not_divide_by_zero():
    flat = np.array([(700.0, -1.3), (700.0, -1.3)])
    assert np.all(np.isfinite(scalarize(flat, EVEN)))


def test_rho_must_be_finite_and_non_negative():
    with np.testing.assert_raises_regex(ValueError, "rho must be finite"):
        scalarize(OBSERVED, EVEN, rho=np.inf)
    pytest.importorskip("quantile_forest")
    with np.testing.assert_raises_regex(ValueError, "rho must be finite"):
        ParEGOQuantileRF(np.arange(12.0)[:, None], np.arange(12), rho=-0.01)


def test_only_the_final_profile_is_accepted():
    assert resolve_parego_profile().profile == "scalarized_leaf3_seeded"
    with pytest.raises(ValueError, match="unknown ParEGO profile"):
        resolve_parego_profile("v3")


def test_degenerate_surface_breaks_ties_with_the_run_seed():
    pytest.importorskip("quantile_forest")

    class FlatForest:
        def __init__(self):
            tree_state = type("TreeState", (), {"max_depth": 0})()
            self.estimators_ = [type("Tree", (), {"tree_": tree_state})()]

        def fit(self, X, y):
            return self

        def predict(self, X, quantiles):
            return np.zeros(len(X))

    def proposal(seed):
        optimizer = ParEGOQuantileRF(np.arange(23.0)[:, None], np.arange(23), random_seed=seed)
        optimizer.rf = FlatForest()
        for gidx in range(20):
            optimizer.tell(gidx, [float(gidx), -float(gidx)], 0.01)
        return optimizer.ask(), optimizer

    proposals = [proposal(seed) for seed in range(10)]
    assert len({selected for selected, _ in proposals}) > 1
    for selected, optimizer in proposals:
        assert selected in {20, 21, 22}
        assert optimizer.last_tie_candidate_count == 3
        assert optimizer.last_surrogate_prediction_degenerate is True
        assert optimizer.last_acquisition_score_degenerate is True
        assert optimizer.last_selection_reason == "degenerate_surface_seeded_random"


def test_portfolio_mean_noise_propagates_scenario_mean_variances():
    assert _portfolio_noise([0.04, 0.16], "mean") == 0.05
    assert _portfolio_noise([0.04, None], "mean") is None
    assert _portfolio_noise([0.04, 0.16], "max") is None
