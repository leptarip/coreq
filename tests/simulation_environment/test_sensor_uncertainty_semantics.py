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
"""Pin the corrected signed-bias and stochastic uncertainty contracts."""

import math

import pytest

from source.simulation_environment.cfg.sensor_cfg import sample_uncertainty

WORST_CASE = "worst_case"
NORMAL = "normal"

# Negative signed-bias levels used by V3.
NEGATIVE_UNC_P_LEVELS = (-1.0, -0.5, -0.2)
NEGATIVE_UNC_V_LEVELS = (-2.0, -1.5, -1.0, -0.5, -0.2)


class TestWorstCaseDeterministic:
    """`worst_case` is a deterministic offset: no sampler, no randomness."""

    @pytest.mark.parametrize("nominal", [0.0, 0.2, 0.5, 1.0, 1.5, 2.0])
    def test_non_negative_values_pass_through_unchanged(self, nominal):
        assert sample_uncertainty(nominal, 0.0, WORST_CASE) == nominal

    @pytest.mark.parametrize("nominal", [0.0, 0.2, 1.0])
    def test_is_deterministic_across_repeated_calls(self, nominal):
        values = {sample_uncertainty(nominal, 0.0, WORST_CASE) for _ in range(20)}
        assert len(values) == 1

    @pytest.mark.parametrize("nominal", NEGATIVE_UNC_P_LEVELS + NEGATIVE_UNC_V_LEVELS)
    def test_negative_values_pass_through_unchanged(self, nominal):
        assert sample_uncertainty(nominal, 0.0, WORST_CASE) == nominal

    def test_every_signed_velocity_level_is_distinct(self):
        distinct = {sample_uncertainty(v, 0.0, WORST_CASE)
                    for v in NEGATIVE_UNC_V_LEVELS + (0.0,)}
        assert distinct == set(NEGATIVE_UNC_V_LEVELS + (0.0,))

    def test_positive_levels_remain_distinct(self):
        """The positive half of the axis is unaffected and must stay that way."""
        positive = (0.2, 0.5, 1.0, 1.5, 2.0)
        assert len({sample_uncertainty(v, 0.0, WORST_CASE) for v in positive}) == len(positive)


class TestNormalMode:
    """`normal` accepts only non-negative standard deviations."""

    def test_zero_sigma_yields_no_perturbation(self):
        assert sample_uncertainty(0.0, 0.0, NORMAL) == 0.0

    @pytest.mark.parametrize("nominal", [-1.5, -0.5, -0.2])
    def test_negative_sigma_is_rejected(self, nominal):
        with pytest.raises(ValueError, match="non-negative nominal"):
            sample_uncertainty(nominal, 0.0, NORMAL)

    def test_positive_sigma_draws_a_random_value(self):
        draws = {sample_uncertainty(1.0, 0.0, NORMAL) for _ in range(50)}
        assert len(draws) > 1, "a positive sigma must actually randomise"

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown sensor uncertainty type"):
            sample_uncertainty(1.0, 0.0, "something_else")


class TestImportanceSamplingBias:
    """The `unc_p_add` / `unc_v_add` path. No shipped config enables it, but the
    guards should stay intact through the semantics change."""

    def test_worst_case_refuses_to_be_importance_sampled(self):
        from source.simulation_environment.configuration_error import (
            SimulationConfigurationError,
        )
        with pytest.raises(SimulationConfigurationError):
            sample_uncertainty(1.0, 0.25, WORST_CASE)

    def test_normal_bias_requires_a_positive_nominal_sigma(self):
        class _Sampler:
            def normal(self, **kwargs):
                return 0.0

        with pytest.raises(ValueError, match="positive nominal and proposal"):
            sample_uncertainty(0.0, 0.25, NORMAL, sampler=_Sampler())

    def test_normal_proposal_requires_sampler(self):
        with pytest.raises(ValueError, match="requires an importance sampler"):
            sample_uncertainty(1.0, 0.25, NORMAL)

    def test_normal_bias_routes_through_the_sampler(self):
        captured = {}

        class _Sampler:
            def normal(self, **kwargs):
                captured.update(kwargs)
                return 0.123

        assert sample_uncertainty(1.0, 0.25, NORMAL, sampler=_Sampler()) == 0.123
        assert captured["sigma_base"] == 1.0
        assert math.isclose(captured["sigma_bias"], 1.25)


def test_v3_signed_uncertainty_levels_are_all_reachable():
    unc_p = (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0)
    unc_v = (-2.0, -1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 1.5, 2.0)

    eff_p = {sample_uncertainty(v, 0.0, WORST_CASE) for v in unc_p}
    eff_v = {sample_uncertainty(v, 0.0, WORST_CASE) for v in unc_v}

    assert len(eff_p) == len(unc_p)
    assert len(eff_v) == len(unc_v)
    assert len(eff_p) * len(eff_v) == 77
