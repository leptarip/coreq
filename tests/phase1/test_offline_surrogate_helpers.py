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

"""Offline calibration, ensemble probabilities and model serialization."""

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from source.design_exploration.surrogates.margin_calibration import (
    compute_gaussian_calibration,
    compute_student_t_calibration,
)
from source.design_exploration.surrogates.nn_cls.ensemble import (
    NNEnsembleCfg,
    _fit_ensemble,
    _predict_proba_components,
    _predict_proba_ensemble,
)
from source.design_exploration.surrogates.nn_cls.offline_train_nn_ensemble import CalibratedEnsemble


@pytest.mark.parametrize("student_t", [False, True])
def test_margin_calibration_recovers_bias_and_target_coverage(student_t):
    mu = np.array([-1.0, 0.0, 2.0])
    residuals = np.linspace(-2.0, 2.0, 100)
    margins = {i: m + 0.4 + residuals for i, m in enumerate(mu)}
    args = (mu, np.ones(3), margins, np.arange(3))
    if student_t:
        result = compute_student_t_calibration(*args, 5.0, 0.8, 40, 0.001, (0.001, 1000.0))
    else:
        result = compute_gaussian_calibration(*args, 0.8, 40, 0.001, (0.001, 1000.0))
    bias, scale, designs, episodes, _, coverage, _, indices = result
    assert bias == pytest.approx(0.4)
    assert scale > 0
    assert (designs, episodes) == (3, 300)
    assert coverage == pytest.approx(0.8, abs=0.001)
    np.testing.assert_array_equal(np.sort(indices), np.arange(3))


@pytest.mark.parametrize("student_t", [False, True])
def test_margin_calibration_without_evidence_is_identity(student_t):
    args = (np.array([]), np.array([]), {}, np.array([], dtype=int))
    if student_t:
        result = compute_student_t_calibration(*args, 5.0, 0.9, 40, 0.001, (0.001, 1000.0))
    else:
        result = compute_gaussian_calibration(*args, 0.9, 40, 0.001, (0.001, 1000.0))
    assert result[:4] == (0.0, 1.0, 0, 0)
    assert np.isnan(result[4:7]).all()
    assert result[7].size == 0


def test_student_t_calibration_requires_finite_variance():
    with pytest.raises(ValueError, match="df must be > 2"):
        compute_student_t_calibration(
            np.zeros(1), np.ones(1), {0: np.array([-1.0, 1.0])},
            np.array([0]), 2.0, 0.9, 40, 0.001, (0.001, 1000.0),
        )



def test_ensemble_preserves_safe_and_unsafe_columns_for_single_class_members():
    X = np.zeros((4, 2))
    models = [DummyClassifier(strategy="constant", constant=c).fit(X, np.full(4, c)) for c in (0, 1)]
    np.testing.assert_array_equal(_predict_proba_ensemble(models, X), np.full((4, 2), 0.5))
    np.testing.assert_array_equal(_predict_proba_components(models, X), [[1.0] * 4, [0.0] * 4])


@pytest.mark.filterwarnings("ignore:Stochastic Optimizer:sklearn.exceptions.ConvergenceWarning")
def test_fitted_offline_ensemble_predicts_after_joblib_roundtrip(tmp_path):
    X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    cfg = NNEnsembleCfg(seed=17, ensemble_size=2, hidden_layers=(4,), max_iter=50,
                        early_stopping=False, batch_size=16)
    model = CalibratedEnsemble(_fit_ensemble(X, (X[:, 0] > 0).astype(int), cfg))
    path = tmp_path / "ensemble.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    probs = restored.predict_proba(X)
    components = restored.predict_component_proba(X)
    np.testing.assert_array_equal(probs, model.predict_proba(X))
    np.testing.assert_array_equal(components, model.predict_component_proba(X))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    np.testing.assert_allclose(probs[:, 0], components.mean(axis=0))
    assert components.shape == (2, len(X))
