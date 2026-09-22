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
"""Pin the per-model behaviour of the network delay and packet-drop laws.

Two contracts matter here. The nominal path must stay deterministic under a
seeded RNG and must not consume randomness it does not need -- an extra draw
shifts every subsequent sample and breaks episode replay. The importance-sampled
path must hand the sampler the *base* law and the *proposal* law as separate
arguments, because the likelihood ratio is computed from the pair; a model that
passes the biased parameters as the base silently reports weight 1.
"""

import math
import random

import numpy as np
import pytest

from source.simulation_environment.communication.network_models import network_models_factory
from source.simulation_environment.communication.network_models.network_log_norm import (
    NetworkLogicLogNorm,
)
from source.simulation_environment.communication.network_models.network_uniform import (
    NetworkLogicUniform,
)
from source.simulation_environment.configuration_error import SimulationConfigurationError


UNIFORM_CFG = {"packet_drop_rate": 0.1, "delay": 10, "jitter": 4}
LOG_NORM_CFG = {"packet_drop_rate": 0.1, "delay_min": 5, "delay_avg": 20, "jitter": 6}


@pytest.fixture(autouse=True)
def _isolate_global_rng():
    """These tests seed the process-wide generators; hand them back untouched so
    the suite stays independent of file ordering."""
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    yield
    random.setstate(random_state)
    np.random.set_state(numpy_state)


class RecordingSampler:
    """Captures the arguments each draw is issued with, without weighting."""

    def __init__(self, *, bernoulli=False, uniform=0.0, shifted_lognormal=0.0):
        self.calls = []
        self._bernoulli = bernoulli
        self._uniform = uniform
        self._shifted_lognormal = shifted_lognormal

    def bernoulli(self, **kwargs):
        self.calls.append(("bernoulli", kwargs))
        return self._bernoulli

    def uniform(self, **kwargs):
        self.calls.append(("uniform", kwargs))
        return self._uniform

    def shifted_lognormal(self, **kwargs):
        self.calls.append(("shifted_lognormal", kwargs))
        return self._shifted_lognormal

    @property
    def last(self):
        return self.calls[-1][1]


class TestUniformValidation:

    def test_a_complete_config_is_accepted(self):
        assert NetworkLogicUniform.validate_cfg(UNIFORM_CFG) == (True, "")

    @pytest.mark.parametrize("missing", ["packet_drop_rate", "delay", "jitter"])
    def test_every_parameter_is_required(self, missing):
        cfg = {k: v for k, v in UNIFORM_CFG.items() if k != missing}

        valid, reason = NetworkLogicUniform.validate_cfg(cfg)

        assert not valid
        assert "Missing parameters" in reason

    @pytest.mark.parametrize("field", ["delay", "jitter"])
    def test_negative_timing_is_rejected(self, field):
        valid, reason = NetworkLogicUniform.validate_cfg({**UNIFORM_CFG, field: -1})

        assert not valid
        assert "must be positive" in reason

    @pytest.mark.parametrize("rate", [-0.1, 1.1, 2.0])
    def test_drop_rate_outside_the_unit_interval_is_rejected(self, rate):
        valid, reason = NetworkLogicUniform.validate_cfg({**UNIFORM_CFG, "packet_drop_rate": rate})

        assert not valid
        assert "between 0 and 1" in reason

    @pytest.mark.parametrize("rate", [0.0, 0.5, 1.0])
    def test_drop_rate_bounds_are_inclusive(self, rate):
        assert NetworkLogicUniform.validate_cfg({**UNIFORM_CFG, "packet_drop_rate": rate})[0]

    @pytest.mark.parametrize("field", ["delay", "jitter"])
    def test_zero_timing_is_allowed(self, field):
        assert NetworkLogicUniform.validate_cfg({**UNIFORM_CFG, field: 0})[0]


class TestUniformDelay:

    def test_zero_jitter_yields_the_configured_delay(self):
        model = NetworkLogicUniform({**UNIFORM_CFG, "jitter": 0})

        assert {model.compute_De2e() for _ in range(20)} == {10}

    def test_delay_stays_inside_the_configured_band(self):
        model = NetworkLogicUniform(UNIFORM_CFG)
        random.seed(4242)

        draws = [model.compute_De2e() for _ in range(500)]

        assert min(draws) >= UNIFORM_CFG["delay"]
        assert max(draws) <= UNIFORM_CFG["delay"] + UNIFORM_CFG["jitter"]
        assert len(set(draws)) > 1, "a positive jitter must actually randomise"

    def test_delay_is_a_whole_number_of_milliseconds(self):
        model = NetworkLogicUniform(UNIFORM_CFG)
        random.seed(11)

        assert all(isinstance(model.compute_De2e(), int) for _ in range(50))

    def test_nominal_delay_is_reproducible_under_a_seed(self):
        model = NetworkLogicUniform(UNIFORM_CFG)

        random.seed(99)
        first = [model.compute_De2e() for _ in range(30)]
        random.seed(99)
        second = [model.compute_De2e() for _ in range(30)]

        assert first == second

    def test_sampler_receives_the_base_band_and_the_proposal_band(self):
        sampler = RecordingSampler(uniform=12.4)
        model = NetworkLogicUniform(
            UNIFORM_CFG, sampler=sampler, bias_cfg={"delay": 50, "jitter": 20}
        )

        model.compute_De2e(sim_time=0)

        assert model.compute_De2e(sim_time=0) == 12  # the drawn value, rounded
        assert sampler.last["a_base"] == 10
        assert sampler.last["b_base"] == 14
        assert sampler.last["a_bias"] == 50
        assert sampler.last["b_bias"] == 70
        assert sampler.last["component"] == "network.delay"

    def test_a_degenerate_proposal_band_is_rejected_by_the_sampler(self):
        """`jitter = 0` collapses the band to a point, which has no density."""
        from source.simulation_environment.rare_events.wrapper import ImportanceSampler

        model = NetworkLogicUniform(
            {**UNIFORM_CFG, "jitter": 0}, sampler=ImportanceSampler(seed=1)
        )

        with pytest.raises(ValueError, match="Bias bounds"):
            model.compute_De2e(sim_time=0)

    def test_bias_without_a_sampler_is_a_configuration_error(self):
        with pytest.raises(SimulationConfigurationError, match="Uniform network bias"):
            NetworkLogicUniform(UNIFORM_CFG, sampler=None, bias_cfg={"delay": 50})


class TestUniformPacketDrop:

    def test_zero_drop_rate_short_circuits_without_consuming_randomness(self):
        """The fast path must not advance the RNG. A stray draw here shifts every
        later sample and breaks seeded replay."""
        model = NetworkLogicUniform({**UNIFORM_CFG, "packet_drop_rate": 0.0})
        random.seed(1234)
        state = random.getstate()

        assert model.should_drop_packet() is False
        assert random.getstate() == state

    def test_a_positive_drop_rate_does_consume_randomness(self):
        model = NetworkLogicUniform({**UNIFORM_CFG, "packet_drop_rate": 0.5})
        random.seed(1234)
        state = random.getstate()

        model.should_drop_packet()

        assert random.getstate() != state

    def test_certain_drop_rate_always_drops(self):
        model = NetworkLogicUniform({**UNIFORM_CFG, "packet_drop_rate": 1.0})
        random.seed(5)

        assert all(model.should_drop_packet() for _ in range(100))

    def test_drop_rate_is_honoured_in_aggregate(self):
        model = NetworkLogicUniform({**UNIFORM_CFG, "packet_drop_rate": 0.3})
        random.seed(20260826)

        drops = sum(model.should_drop_packet() for _ in range(20000))

        assert 0.28 < drops / 20000 < 0.32

    def test_sampler_receives_base_and_proposal_probabilities(self):
        sampler = RecordingSampler(bernoulli=True)
        model = NetworkLogicUniform(
            UNIFORM_CFG, sampler=sampler, bias_cfg={"packet_drop_rate": 0.9}
        )

        assert model.should_drop_packet(sim_time=0) is True
        assert sampler.last["p_base"] == 0.1
        assert sampler.last["p_bias"] == 0.9
        assert sampler.last["component"] == "network.packet_drop"

    def test_the_zero_rate_fast_path_is_skipped_when_the_proposal_is_positive(self):
        """A zero base rate with a positive proposal still needs a weighted draw."""
        sampler = RecordingSampler(bernoulli=True)
        model = NetworkLogicUniform(
            {**UNIFORM_CFG, "packet_drop_rate": 0.0},
            sampler=sampler,
            bias_cfg={"packet_drop_rate": 0.5},
        )

        model.should_drop_packet(sim_time=0)

        assert sampler.calls, "the proposal must be sampled, not short-circuited"
        assert sampler.last["p_base"] == 0.0
        assert sampler.last["p_bias"] == 0.5


class TestLogNormValidation:

    def test_a_complete_config_is_accepted(self):
        assert NetworkLogicLogNorm.validate_cfg(LOG_NORM_CFG) == (True, "")

    @pytest.mark.parametrize("missing", ["packet_drop_rate", "delay_min", "delay_avg", "jitter"])
    def test_every_parameter_is_required(self, missing):
        cfg = {k: v for k, v in LOG_NORM_CFG.items() if k != missing}

        valid, reason = NetworkLogicLogNorm.validate_cfg(cfg)

        assert not valid
        assert "Missing parameters" in reason

    @pytest.mark.parametrize("field", ["delay_min", "delay_avg", "jitter"])
    def test_negative_timing_is_rejected(self, field):
        valid, reason = NetworkLogicLogNorm.validate_cfg({**LOG_NORM_CFG, field: -1})

        assert not valid
        assert "must be >0" in reason

    def test_an_average_below_the_floor_is_rejected(self):
        valid, reason = NetworkLogicLogNorm.validate_cfg(
            {**LOG_NORM_CFG, "delay_min": 30, "delay_avg": 20}
        )

        assert not valid
        assert "delay_min" in reason

    @pytest.mark.parametrize("rate", [-0.1, 1.1])
    def test_drop_rate_outside_the_unit_interval_is_rejected(self, rate):
        valid, reason = NetworkLogicLogNorm.validate_cfg(
            {**LOG_NORM_CFG, "packet_drop_rate": rate}
        )

        assert not valid
        assert "between 0 and 1" in reason

    def test_validator_accepts_a_degenerate_band_the_constructor_rejects(self):
        """Characterisation test for a gap between the two.

        `validate_cfg` only rejects `delay_avg < delay_min`, so an equal pair
        passes validation and then divides by zero while fitting the log-normal
        moments. Tighten the validator to `<=` and this test should be updated
        to expect a rejection.
        """
        degenerate = {**LOG_NORM_CFG, "delay_min": 10, "delay_avg": 10}

        assert NetworkLogicLogNorm.validate_cfg(degenerate) == (True, "")
        with pytest.raises(ZeroDivisionError):
            NetworkLogicLogNorm(degenerate)


class TestLogNormMomentFit:

    @pytest.mark.parametrize(
        "mean,std",
        [(15.0, 6.0), (1.0, 0.25), (100.0, 5.0), (3.5, 12.0)],
    )
    def test_fit_recovers_the_requested_mean_and_standard_deviation(self, mean, std):
        """`_fit_lognormal_params` is moment matching; inverting it must return
        the inputs, or the configured delay distribution is not the one the
        config asked for."""
        mu, sigma = NetworkLogicLogNorm._fit_lognormal_params(mean, std)

        recovered_mean = math.exp(mu + sigma ** 2 / 2.0)
        recovered_var = (math.exp(sigma ** 2) - 1.0) * math.exp(2.0 * mu + sigma ** 2)

        assert recovered_mean == pytest.approx(mean)
        assert math.sqrt(recovered_var) == pytest.approx(std)

    def test_the_fit_is_applied_to_the_variable_part_above_the_floor(self):
        model = NetworkLogicLogNorm(LOG_NORM_CFG)
        expected = NetworkLogicLogNorm._fit_lognormal_params(
            LOG_NORM_CFG["delay_avg"] - LOG_NORM_CFG["delay_min"], LOG_NORM_CFG["jitter"]
        )

        assert (model.mu_base, model.sigma_base) == expected

    def test_base_and_proposal_fits_are_derived_independently(self):
        model = NetworkLogicLogNorm(
            LOG_NORM_CFG,
            sampler=RecordingSampler(),
            bias_cfg={"delay_min": 5, "delay_avg": 60, "jitter": 10},
        )

        assert (model.mu_bias, model.sigma_bias) != (model.mu_base, model.sigma_base)
        assert (model.mu_base, model.sigma_base) == NetworkLogicLogNorm._fit_lognormal_params(
            15, 6
        )


class TestLogNormDelay:

    def test_draws_never_fall_below_the_configured_floor(self):
        model = NetworkLogicLogNorm(LOG_NORM_CFG)
        np.random.seed(0)

        draws = [model.compute_De2e() for _ in range(500)]

        assert min(draws) >= LOG_NORM_CFG["delay_min"]
        assert len(set(draws)) > 1

    def test_delay_is_a_whole_number_of_milliseconds(self):
        model = NetworkLogicLogNorm(LOG_NORM_CFG)
        np.random.seed(3)

        assert all(isinstance(model.compute_De2e(), int) for _ in range(50))

    def test_nominal_delay_is_reproducible_under_a_seed(self):
        model = NetworkLogicLogNorm(LOG_NORM_CFG)

        np.random.seed(7)
        first = [model.compute_De2e() for _ in range(30)]
        np.random.seed(7)
        second = [model.compute_De2e() for _ in range(30)]

        assert first == second

    def test_sampler_receives_shift_mu_and_sigma_for_both_laws(self):
        sampler = RecordingSampler(shifted_lognormal=33.6)
        model = NetworkLogicLogNorm(
            LOG_NORM_CFG, sampler=sampler, bias_cfg={"delay_min": 5, "delay_avg": 60, "jitter": 10}
        )

        assert model.compute_De2e(sim_time=0) == 34  # the drawn value, rounded

        call = sampler.last
        assert call["shift_base"] == 5
        assert (call["mu_base"], call["sigma_base"]) == (model.mu_base, model.sigma_base)
        assert (call["mu_bias"], call["sigma_bias"]) == (model.mu_bias, model.sigma_bias)
        assert call["component"] == "network.delay"

    def test_bias_without_a_sampler_is_a_configuration_error(self):
        with pytest.raises(SimulationConfigurationError, match="Log-normal network bias"):
            NetworkLogicLogNorm(LOG_NORM_CFG, sampler=None, bias_cfg={"delay_avg": 60})


class TestLogNormPacketDrop:

    def test_zero_drop_rate_short_circuits_without_consuming_randomness(self):
        model = NetworkLogicLogNorm({**LOG_NORM_CFG, "packet_drop_rate": 0.0})
        random.seed(77)
        state = random.getstate()

        assert model.should_drop_packet() is False
        assert random.getstate() == state

    def test_drop_rate_is_honoured_in_aggregate(self):
        model = NetworkLogicLogNorm({**LOG_NORM_CFG, "packet_drop_rate": 0.25})
        random.seed(20260826)

        drops = sum(model.should_drop_packet() for _ in range(20000))

        assert 0.23 < drops / 20000 < 0.27

    def test_sampler_receives_base_and_proposal_probabilities(self):
        sampler = RecordingSampler(bernoulli=True)
        model = NetworkLogicLogNorm(
            LOG_NORM_CFG, sampler=sampler, bias_cfg={"packet_drop_rate": 0.75}
        )

        assert model.should_drop_packet(sim_time=0) is True
        assert sampler.last["p_base"] == 0.1
        assert sampler.last["p_bias"] == 0.75


def _windowed(model_class, cfg, sampler, bias):
    window = {"start_ms": 1000, "end_ms": 2000}
    return model_class(cfg, sampler=sampler, bias_cfg={**bias, "active_window": window})


MODELS = [
    pytest.param(NetworkLogicUniform, UNIFORM_CFG, {"packet_drop_rate": 0.9, "delay": 50, "jitter": 20}, id="uniform"),
    pytest.param(NetworkLogicLogNorm, LOG_NORM_CFG, {"packet_drop_rate": 0.9, "delay_min": 5, "delay_avg": 60, "jitter": 10}, id="log_norm"),
]


class TestConditionalBiasWindow:
    """A windowed proposal must be nominal outside the window, and it must say
    so in the component label -- the label is how the weight decomposition
    attributes the ratio."""

    @pytest.mark.parametrize("model_class,cfg,bias", MODELS)
    @pytest.mark.parametrize("sim_time", [0, 999, 2001, 10000])
    def test_drop_proposal_is_nominal_outside_the_window(self, model_class, cfg, bias, sim_time):
        sampler = RecordingSampler(bernoulli=False)
        model = _windowed(model_class, cfg, sampler, bias)

        model.should_drop_packet(sim_time=sim_time)

        assert sampler.last["p_bias"] == sampler.last["p_base"]
        assert sampler.last["component"] == "network.packet_drop"

    @pytest.mark.parametrize("model_class,cfg,bias", MODELS)
    @pytest.mark.parametrize("sim_time", [1000, 1500, 2000])
    def test_drop_proposal_is_biased_inside_the_window(self, model_class, cfg, bias, sim_time):
        sampler = RecordingSampler(bernoulli=False)
        model = _windowed(model_class, cfg, sampler, bias)

        model.should_drop_packet(sim_time=sim_time)

        assert sampler.last["p_bias"] == 0.9
        assert sampler.last["p_base"] == cfg["packet_drop_rate"]
        assert sampler.last["component"] == "network.packet_drop.window"

    @pytest.mark.parametrize("model_class,cfg,bias", MODELS)
    def test_delay_proposal_follows_the_same_window(self, model_class, cfg, bias):
        sampler = RecordingSampler(uniform=55.0, shifted_lognormal=55.0)
        model = _windowed(model_class, cfg, sampler, bias)

        model.compute_De2e(sim_time=500)
        assert sampler.last["component"] == "network.delay"
        outside = dict(sampler.last)

        model.compute_De2e(sim_time=1500)
        assert sampler.last["component"] == "network.delay.window"
        assert sampler.last != outside, "the proposal must actually change inside the window"

    @pytest.mark.parametrize("model_class,cfg,bias", MODELS)
    def test_the_success_budget_retires_the_window(self, model_class, cfg, bias):
        """`until_successes` caps how many non-drops the conditional proposal
        covers; afterwards the draw reverts to the base law."""
        sampler = RecordingSampler(bernoulli=False)  # every draw is a "success"
        model = model_class(
            cfg,
            sampler=sampler,
            bias_cfg={**bias, "active_window": {"start_ms": 1000, "end_ms": 2000,
                                                "until_successes": 2}},
        )

        components = []
        for _ in range(4):
            model.should_drop_packet(sim_time=1500)
            components.append(sampler.last["component"])

        assert components == [
            "network.packet_drop.window",
            "network.packet_drop.window",
            "network.packet_drop",
            "network.packet_drop",
        ]

    @pytest.mark.parametrize("model_class,cfg,bias", MODELS)
    def test_a_drop_does_not_spend_the_success_budget(self, model_class, cfg, bias):
        sampler = RecordingSampler(bernoulli=True)  # every draw is a drop
        model = model_class(
            cfg,
            sampler=sampler,
            bias_cfg={**bias, "active_window": {"start_ms": 1000, "end_ms": 2000,
                                                "until_successes": 1}},
        )

        for _ in range(5):
            model.should_drop_packet(sim_time=1500)

        assert model._window_successes == 0
        assert sampler.last["component"] == "network.packet_drop.window"

    @pytest.mark.parametrize("model_class,cfg,bias", MODELS)
    def test_a_windowed_proposal_requires_a_simulator_time(self, model_class, cfg, bias):
        sampler = RecordingSampler()
        model = _windowed(model_class, cfg, sampler, bias)

        with pytest.raises(ValueError, match="sim_time is required"):
            model.should_drop_packet(sim_time=None)

    @pytest.mark.parametrize("model_class,cfg,bias", MODELS)
    @pytest.mark.parametrize(
        "window", [{"start_ms": -1, "end_ms": 10}, {"start_ms": 20, "end_ms": 10}]
    )
    def test_a_malformed_window_is_rejected_at_construction(self, model_class, cfg, bias, window):
        with pytest.raises(ValueError, match="active_window"):
            model_class(cfg, sampler=RecordingSampler(), bias_cfg={**bias, "active_window": window})


class TestFactory:

    def test_uniform_type_yields_the_uniform_model(self):
        model = network_models_factory.get_network_model({"type": "uniform", **UNIFORM_CFG})

        assert isinstance(model, NetworkLogicUniform)

    def test_log_norm_type_yields_the_log_norm_model(self):
        model = network_models_factory.get_network_model({"type": "log_norm", **LOG_NORM_CFG})

        assert isinstance(model, NetworkLogicLogNorm)

    def test_an_unknown_type_yields_no_model(self):
        assert network_models_factory.get_network_model({"type": "carrier-pigeon"}) is None

    def test_sampler_and_bias_reach_the_constructed_model(self):
        sampler = RecordingSampler()

        model = network_models_factory.get_network_model(
            {"type": "uniform", **UNIFORM_CFG},
            sampler=sampler,
            bias_cfg={"packet_drop_rate": 0.6},
        )

        assert model.sampler is sampler
        assert model.drop_rate_bias == 0.6

    def test_validate_cfg_rejects_an_unknown_type(self):
        valid, reason = network_models_factory.validate_cfg({"network": {"type": "smoke-signal"}})

        assert not valid
        assert reason == "Unknown network type"

    @pytest.mark.parametrize(
        "network",
        [{"type": "uniform", **UNIFORM_CFG}, {"type": "log_norm", **LOG_NORM_CFG}],
    )
    def test_validate_cfg_delegates_to_the_selected_model(self, network):
        assert network_models_factory.validate_cfg({"network": network}) == (True, "")

    def test_validate_cfg_surfaces_the_model_diagnostic(self):
        valid, reason = network_models_factory.validate_cfg(
            {"network": {"type": "uniform", **UNIFORM_CFG, "delay": -5}}
        )

        assert not valid
        assert "NetworkUniform" in reason


class TestRareEventConfigValidation:

    UNIFORM_COMM = {"network": {"type": "uniform", **UNIFORM_CFG}}
    LOG_NORM_COMM = {"network": {"type": "log_norm", **LOG_NORM_CFG}}

    def test_an_absent_bias_section_is_accepted(self):
        assert network_models_factory.validate_rare_event_cfg({}, self.UNIFORM_COMM) == (True, "")

    def test_a_bias_section_without_a_communication_block_is_not_checked(self):
        assert network_models_factory.validate_rare_event_cfg(
            {"network_bias": {"delay": -1}}, None
        ) == (True, "")

    @pytest.mark.parametrize(
        "bias,fragment",
        [
            ({"delay": -1}, "delay must be >= 0"),
            ({"jitter": -1}, "jitter must be >= 0"),
            ({"packet_drop_rate": 1.5}, "packet_drop_rate must be in [0,1]"),
            ({"packet_drop_rate": -0.5}, "packet_drop_rate must be in [0,1]"),
        ],
    )
    def test_uniform_bias_bounds_are_enforced(self, bias, fragment):
        valid, reason = network_models_factory.validate_rare_event_cfg(
            {"network_bias": bias}, self.UNIFORM_COMM
        )

        assert not valid
        assert fragment in reason

    @pytest.mark.parametrize(
        "bias,fragment",
        [
            ({"delay_min": -1}, "delay_min must be >= 0"),
            ({"delay_avg": -1}, "delay_avg must be >= 0"),
            ({"delay_min": 10, "delay_avg": 5}, "delay_avg must be >= delay_min"),
            ({"jitter": -1}, "jitter must be >= 0"),
            ({"packet_drop_rate": 2}, "packet_drop_rate must be in [0,1]"),
        ],
    )
    def test_log_norm_bias_bounds_are_enforced(self, bias, fragment):
        valid, reason = network_models_factory.validate_rare_event_cfg(
            {"network_bias": bias}, self.LOG_NORM_COMM
        )

        assert not valid
        assert fragment in reason

    @pytest.mark.parametrize(
        "bias",
        [
            {"delay": 50, "jitter": 20},
            {"packet_drop_rate": 0.0},
            {"packet_drop_rate": 1.0},
            {"active_window": {"start_ms": 0, "end_ms": 100}, "delay": 50},
        ],
    )
    def test_valid_uniform_bias_is_accepted(self, bias):
        assert network_models_factory.validate_rare_event_cfg(
            {"network_bias": bias}, self.UNIFORM_COMM
        ) == (True, "")

    def test_an_unsupported_network_type_cannot_be_bias_validated(self):
        valid, reason = network_models_factory.validate_rare_event_cfg(
            {"network_bias": {"delay": 1}}, {"network": {"type": "smoke-signal"}}
        )

        assert not valid
        assert "not supported" in reason

    @pytest.mark.parametrize(
        "window",
        [
            {"start_ms": 10},
            {"end_ms": 10},
            {"start_ms": 20, "end_ms": 10},
            {"start_ms": -1, "end_ms": 10},
            {"start_ms": 0, "end_ms": 10, "until_successes": 0},
            {"start_ms": 0, "end_ms": 10, "extra": 1},
        ],
    )
    def test_a_malformed_window_is_rejected(self, window):
        valid, reason = network_models_factory.validate_rare_event_cfg(
            {"network_bias": {"active_window": window}}, self.UNIFORM_COMM
        )

        assert not valid
        assert "active_window" in reason
