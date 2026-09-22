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

"""QRF probability calibration must preserve ordering across the tail boundary."""
import numpy as np
import pytest

pytest.importorskip("quantile_forest")

from source.design_exploration.commons import surrogate_loading
from source.design_exploration.surrogates.qrf import predict_qrf_offline, train_qrf_offline


def test_fitted_calibration_preserves_order_and_low_risk_scores(monkeypatch):
    design_ids = np.repeat(np.arange(4), 20)
    X = design_ids[:, None].astype(float)
    margins = np.ones(80)
    for i, violations in enumerate([0, 1, 2, 4]):
        margins[i * 20:i * 20 + violations] = -1.0
    raw = np.array([0.001, 0.04, 0.1, 0.2])
    monkeypatch.setattr(
        train_qrf_offline, "_predict_p_unsafe",
        lambda model, X, taus: raw[X[:, 0].astype(int)],
    )
    calibration = train_qrf_offline._fit_p_calibrator(
        None, X, margins, design_ids, np.ones(80, dtype=bool), [0.001, 0.999],
    )
    probes = np.array([0.001, 0.01, 0.04, 0.099999, 0.1, 0.100001, 0.2, 0.9])
    expected = np.maximum(
        probes, np.interp(probes, calibration["x"], calibration["y"]),
    )
    for apply_calibration in (
        train_qrf_offline._apply_p_calibration,
        predict_qrf_offline._apply_p_calibration,
        surrogate_loading._apply_p_calibration,
    ):
        scores = apply_calibration(probes, calibration)
        np.testing.assert_allclose(scores, expected)
        assert np.all(np.diff(scores) >= 0)
        assert scores[0] < 0.05
        assert scores[1] > probes[1]  # Isotonic calibration still applies.
        assert scores[-1] == probes[-1]  # Upper envelope still applies.
