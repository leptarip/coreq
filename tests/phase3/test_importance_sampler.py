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

import copy
import math
import random
from types import SimpleNamespace

import pytest

from source.simulation_environment.configuration_error import SimulationConfigurationError
from source.simulation_environment.communication.network_models.network_log_norm import (
    NetworkLogicLogNorm,
)
from source.simulation_environment.communication.network_models.network_uniform import (
    NetworkLogicUniform,
)
from source.simulation_environment.communication.fault_models.sit import SITLogic
from source.simulation_environment.obps.logic.sector_logic import SectorLogic
from source.simulation_environment.rare_events.wrapper import (
    ImportanceSampler,
    defensive_mixture_log_weight,
    finite_mixture_log_weight,
    replay_configuration_is_nominal,
    _logpdf_lognormal,
    _sample_categorical,
    require_sampler_for_bias,
)


def test_bernoulli_weight_is_base_over_proposal():
    sampler = ImportanceSampler(seed=1)

    outcome = sampler.bernoulli(p_base=0.2, p_bias=0.8)

    assert outcome is True
    assert sampler.log_weight == pytest.approx(math.log(0.2 / 0.8))
    assert sampler.weight() == pytest.approx(0.25)
    assert sampler.draw_count == 1


def test_episode_seed_reset_is_reproducible_and_clears_weight():
    sampler = ImportanceSampler(seed=999)
    first = [sampler.bernoulli(0.3, 0.6) for _ in range(8)]

    sampler.reset(seed=999)
    second = [sampler.bernoulli(0.3, 0.6) for _ in range(8)]

    assert first == second
    assert sampler.draw_count == 8
    assert sampler.component_log_ratios == {"unlabeled": pytest.approx(sampler.component_log_ratio)}

    sampler.reset(seed=1000)
    assert sampler.component_log_ratio == 0.0
    assert sampler.component_log_ratios == {}
    assert sampler.component_draw_counts == {}


def test_named_component_contributions_sum_to_total_and_replay():
    sampler = ImportanceSampler(seed=17, defensive_mixture_probability=0.2)

    def draw_episode():
        sampler.categorical(
            [0.8, 0.2], [0.3, 0.7], component="sensor.outcome"
        )
        sampler.shifted_lognormal(
            50.0, 3.0, 0.5, 50.0, 3.3, 0.7,
            component="network.delay",
        )
        sampler.bernoulli(0.01, 0.13, component="network.packet_drop")
        sampler.normal(3000.0, 1000.0, 2800.0, 900.0, component="fault.trigger")
        return (
            sampler.component_log_ratio,
            sampler.component_log_ratios,
            sampler.component_draw_counts,
            sampler.log_weight,
        )

    first = draw_episode()
    assert set(first[1]) == {
        "sensor.outcome", "network.delay", "network.packet_drop", "fault.trigger"
    }
    assert sum(first[1].values()) == pytest.approx(first[0])
    assert first[2] == {name: 1 for name in first[1]}

    sampler.reset(seed=17)
    second = draw_episode()
    assert second[0] == pytest.approx(first[0])
    assert second[1] == pytest.approx(first[1])
    assert second[2] == first[2]
    assert second[3] == pytest.approx(first[3])


def test_identity_proposal_has_unit_weight_with_named_components():
    sampler = ImportanceSampler(seed=21)
    sampler.categorical([0.8, 0.2], [0.8, 0.2], component="sensor.outcome")
    sampler.bernoulli(0.01, 0.01, component="network.packet_drop")
    sampler.normal(0.0, 1.0, 0.0, 1.0, component="fault.trigger")

    assert sampler.component_log_ratio == pytest.approx(0.0)
    assert all(value == pytest.approx(0.0) for value in sampler.component_log_ratios.values())
    assert sampler.weight() == pytest.approx(1.0)


def test_recorded_nominal_prefix_replays_exactly_then_branches():
    parent = ImportanceSampler(seed=41, record_draw_history=True)
    parent.set_context_time_ms(2500)
    first = parent.categorical([0.75, 0.25], [0.75, 0.25], component="sensor.outcome")
    parent.set_context_time_ms(2510)
    second = parent.shifted_lognormal(
        50.0, 3.0, 0.5, 50.0, 3.0, 0.5, component="network.delay"
    )
    prefix = parent.draw_history

    child = ImportanceSampler(seed=99, record_draw_history=True)
    child.reset(seed=99, replay_prefix=prefix, replay_under_nominal_law=True)
    child.set_context_time_ms(2500)
    assert child.categorical(
        [0.75, 0.25], [0.75, 0.25], component="sensor.outcome"
    ) == first
    child.set_context_time_ms(2510)
    assert child.shifted_lognormal(
        50.0, 3.0, 0.5, 50.0, 3.0, 0.5, component="network.delay"
    ) == second
    assert child.replay_prefix_consumed
    assert [value["source"] for value in child.draw_history] == ["replayed", "replayed"]

    child.set_context_time_ms(2550)
    child.bernoulli(0.01, 0.01, component="network.packet_drop")
    assert child.draw_history[-1]["source"] == "fresh"
    assert child.component_log_ratio == pytest.approx(0.0)


def test_replay_rejects_a_different_stochastic_call_sequence():
    parent = ImportanceSampler(seed=51, record_draw_history=True)
    parent.set_context_time_ms(2600)
    parent.bernoulli(0.1, 0.1, component="network.packet_drop")

    child = ImportanceSampler(seed=52, record_draw_history=True)
    child.reset(
        seed=52, replay_prefix=parent.draw_history, replay_under_nominal_law=True
    )
    child.set_context_time_ms(2650)
    with pytest.raises(SimulationConfigurationError, match="Stochastic replay diverged"):
        child.bernoulli(0.1, 0.1, component="network.packet_drop")


def test_replay_rejects_importance_biased_distributions():
    parent = ImportanceSampler(seed=61, record_draw_history=True)
    parent.bernoulli(0.1, 0.1, component="network.packet_drop")

    child = ImportanceSampler(seed=62, record_draw_history=True)
    child.reset(
        seed=62, replay_prefix=parent.draw_history, replay_under_nominal_law=True
    )
    with pytest.raises(SimulationConfigurationError, match="only under the nominal law"):
        child.bernoulli(0.1, 0.2, component="network.packet_drop")


def test_replay_requires_explicit_nominal_law_declaration():
    parent = ImportanceSampler(seed=71, record_draw_history=True)
    parent.bernoulli(0.1, 0.1, component="network.packet_drop")

    child = ImportanceSampler(seed=72, record_draw_history=True)
    with pytest.raises(SimulationConfigurationError, match="nominal-law declaration"):
        child.reset(seed=72, replay_prefix=parent.draw_history)


def test_replay_nominal_declaration_is_verified_against_simulator_parameters():
    config = {
        "communication": {
            "network": {
                "type": "log_norm", "delay_min": 50.0, "delay_avg": 100.0,
                "jitter": 20.0, "packet_drop_rate": 0.1,
            },
            "fault": {"trigger_mean": 3000.0, "trigger_var": 1000.0},
        },
    }
    nominal = {
        "sensor_bias": {"miss_scale": 1.0},
        "network_bias": {
            "delay_min": 50.0, "delay_avg": 100.0, "jitter": 20.0,
            "packet_drop_rate": 0.1,
        },
        "fault_bias": {"trigger_mean_add": 0.0, "trigger_var_scale": 1.0},
    }

    assert replay_configuration_is_nominal(config, nominal) is True
    biased = copy.deepcopy(nominal)
    biased["sensor_bias"]["miss_scale"] = 2.0
    assert replay_configuration_is_nominal(config, biased) is False


def test_conflicting_infinite_contributions_are_rejected_before_nan_escapes():
    sampler = ImportanceSampler(seed=73)
    sampler.incorporate_logpdf(math.inf, 0.0, component="outside.bias.support")

    with pytest.raises(SimulationConfigurationError, match="became NaN"):
        sampler.incorporate_logpdf(0.0, math.inf, component="zero.nominal.mass")

    assert sampler.component_log_ratio == math.inf
    assert not math.isnan(sampler.log_weight)


def test_packet_proposal_is_nominal_outside_conditioning_window():
    sampler = ImportanceSampler(seed=23)
    network = NetworkLogicUniform(
        {"packet_drop_rate": 0.1, "delay": 50.0, "jitter": 10.0},
        sampler=sampler,
        bias_cfg={
            "packet_drop_rate": 0.8,
            "delay": 50.0,
            "jitter": 10.0,
            "active_window": {"start_ms": 100, "end_ms": 200},
        },
    )

    network.should_drop_packet(sim_time=50)
    before = sampler.component_log_ratio
    network.should_drop_packet(sim_time=150)
    inside = sampler.component_log_ratio
    network.should_drop_packet(sim_time=250)

    assert before == pytest.approx(0.0)
    assert inside != pytest.approx(0.0)
    assert sampler.component_log_ratio == pytest.approx(inside)
    assert sampler.component_draw_counts["network.packet_drop"] == 2
    assert sampler.component_draw_counts["network.packet_drop.window"] == 1


def test_windowed_sensor_uses_conditional_likelihood_only_inside_window(monkeypatch):
    sampler = ImportanceSampler(seed=31)
    sensor_cfg = SimpleNamespace(
        position=(0.0, 0.0, 0.0), orientation=0.0, fov_degrees=120,
        radius=20.0, probA_det=0.8, probA_wrong=0.0, probA_miss=0.2,
        probA_ghost=0.0, probB_det=0.8, probB_wrong=0.0, probB_miss=0.2,
        probB_ghost=0.0, unc_pos=0.0, unc_vel=0.0, unc_type="normal",
    )
    sensor = SectorLogic(
        sensor_cfg,
        sensor_bias={
            "miss_scale": 4.0,
            "active_window": {"start_ms": 100, "end_ms": 200},
        },
        sampler=sampler,
    )

    class AlwaysInside:
        @staticmethod
        def covers(_shape):
            return True

        @staticmethod
        def intersects(_shape):
            return False

    class ShapeMap:
        clear = staticmethod(lambda: None)
        load = staticmethod(lambda _shapes: None)
        get_shape_from_actor = staticmethod(lambda _actor: object())

    sensor.fov_polygon_A = AlwaysInside()
    sensor.fov_polygon_B = AlwaysInside()
    sensor.actor_sh_map = ShapeMap()
    monkeypatch.setattr(sensor, "compute_result_A", lambda poll, _target, _shape: poll)

    sensor.logic_step(object(), [], sim_time=50)
    before = sampler.component_log_ratio
    sensor.logic_step(object(), [], sim_time=150)
    inside = sampler.component_log_ratio
    sensor.logic_step(object(), [], sim_time=250)

    assert before == pytest.approx(0.0)
    assert inside != pytest.approx(0.0)
    assert sampler.component_log_ratio == pytest.approx(inside)
    assert sampler.component_draw_counts["sensor.outcome"] == 2
    assert sampler.component_draw_counts["sensor.outcome.window"] == 1


def test_windowed_sensor_cross_evaluates_mild_and_strong_miss_scales(monkeypatch):
    sampler = ImportanceSampler(seed=31, audit_proposal_ids=("mild", "strong"))
    sensor_cfg = SimpleNamespace(
        position=(0.0, 0.0, 0.0), orientation=0.0, fov_degrees=120,
        radius=20.0, probA_det=0.7, probA_wrong=0.0, probA_miss=0.3,
        probA_ghost=0.0, probB_det=0.7, probB_wrong=0.0, probB_miss=0.3,
        probB_ghost=0.0, unc_pos=0.0, unc_vel=0.0, unc_type="normal",
    )
    sensor = SectorLogic(
        sensor_cfg,
        sensor_bias={
            "miss_scale": 6.0,
            "audit_miss_scales": {"mild": 6.0, "strong": 12.0},
            "active_window": {"start_ms": 100, "end_ms": 200, "until_successes": 1},
        },
        sampler=sampler,
    )

    class AlwaysInside:
        covers = staticmethod(lambda _shape: True)
        intersects = staticmethod(lambda _shape: False)

    sensor.fov_polygon_A = AlwaysInside()
    sensor.fov_polygon_B = AlwaysInside()
    monkeypatch.setattr(sensor, "compute_result_A", lambda poll, _target, _shape: poll)
    target = SimpleNamespace(id=17)
    sensor.logic_step(target, [], sim_time=150, shape_by_actor_id={17: object()})

    assert sampler.audit_proposal_log_ratios["mild"] == pytest.approx(
        sampler.component_log_ratio
    )
    assert sampler.audit_proposal_log_ratios["strong"] != pytest.approx(
        sampler.component_log_ratio
    )


def test_sensor_network_drop_and_fault_share_one_likelihood_accumulator(monkeypatch):
    sampler = ImportanceSampler(seed=31)
    network = NetworkLogicLogNorm(
        {
            "packet_drop_rate": 0.01,
            "delay_min": 50.0,
            "delay_avg": 100.0,
            "jitter": 20.0,
        },
        sampler=sampler,
        bias_cfg={
            "packet_drop_rate": 0.13,
            "delay_min": 50.0,
            "delay_avg": 220.0,
            "jitter": 60.0,
        },
    )
    fault = SITLogic(
        0.05,
        {"offline": 1500, "trigger_mean": 3000.0, "trigger_var": 1000.0},
        sampler=sampler,
        bias_cfg={"trigger_mean_add": -100.0, "trigger_var_scale": 1.0},
    )
    sensor_cfg = SimpleNamespace(
        position=(0.0, 0.0, 0.0), orientation=0.0, fov_degrees=120,
        radius=20.0, probA_det=0.99, probA_wrong=0.0, probA_miss=0.01,
        probA_ghost=0.0, probB_det=0.8, probB_wrong=0.0, probB_miss=0.2,
        probB_ghost=0.0, unc_pos=0.5, unc_vel=1.0, unc_type="normal",
    )
    sensor = SectorLogic(
        sensor_cfg, sensor_bias={"miss_scale": 5.0}, sampler=sampler
    )

    class AlwaysInside:
        @staticmethod
        def covers(_shape):
            return True

        @staticmethod
        def intersects(_shape):
            return False

    class ShapeMap:
        @staticmethod
        def clear():
            return None

        @staticmethod
        def load(_shapes):
            return None

        @staticmethod
        def get_shape_from_actor(_actor):
            return object()

    sensor.fov_polygon_A = AlwaysInside()
    sensor.fov_polygon_B = AlwaysInside()
    sensor.actor_sh_map = ShapeMap()
    monkeypatch.setattr(sensor, "compute_result_A", lambda poll, _target, _shape: poll)

    network.compute_De2e()
    network.should_drop_packet()
    sensor.logic_step(object(), [])

    assert network.sampler is sampler
    assert fault.sampler is sampler
    assert sensor.sampler is sampler
    assert {
        "fault.trigger", "network.delay", "network.packet_drop", "sensor.outcome"
    }.issubset(sampler.component_log_ratios)
    assert sum(sampler.component_log_ratios.values()) == pytest.approx(
        sampler.component_log_ratio
    )


def test_defensive_episode_mixture_uses_exact_bounded_likelihood_ratio():
    alpha = 0.2
    sampler = ImportanceSampler(seed=1, defensive_mixture_probability=alpha)
    assert sampler.proposal_component == "nominal"
    outcome = sampler.bernoulli(p_base=0.2, p_bias=0.8)
    component_ratio = (0.2 / 0.8) if outcome else (0.8 / 0.2)
    expected = component_ratio / (alpha * component_ratio + 1.0 - alpha)
    assert sampler.component_log_ratio == pytest.approx(math.log(component_ratio))
    assert sampler.weight() == pytest.approx(expected)
    assert sampler.weight() <= 1.0 / alpha

    sampler.reset(seed=1)
    assert sampler.proposal_component == "nominal"
    assert sampler.bernoulli(0.2, 0.8) is outcome


def test_defensive_mixture_transform_is_stable_at_extreme_log_ratios():
    assert defensive_mixture_log_weight(math.inf, 0.2) == pytest.approx(math.log(5.0))
    assert defensive_mixture_log_weight(-math.inf, 0.2) == -math.inf
    assert defensive_mixture_log_weight(1000.0, 0.2) == pytest.approx(math.log(5.0))


def test_finite_mixture_weight_matches_direct_density_sum():
    ratios = {"nominal": 0.0, "mild": math.log(0.3 / 0.5), "strong": math.log(0.3 / 0.8)}
    probabilities = {"nominal": 0.1, "mild": 0.45, "strong": 0.45}
    expected = 0.3 / (0.1 * 0.3 + 0.45 * 0.5 + 0.45 * 0.8)

    assert math.exp(finite_mixture_log_weight(ratios, probabilities)) == pytest.approx(expected)


def test_categorical_cross_evaluates_multiple_proposals_exactly(monkeypatch):
    sampler = ImportanceSampler(
        seed=1, audit_proposal_ids=("mild", "strong")
    )
    monkeypatch.setattr(sampler.rng, "random", lambda: 0.99)

    outcome = sampler.categorical(
        [0.7, 0.3],
        [0.28, 0.72],
        component="sensor.outcome.window",
        audit_probs_bias={
            "mild": [0.4375, 0.5625],
            "strong": [0.16279069767441862, 0.8372093023255814],
        },
    )

    assert outcome == 1
    assert sampler.audit_proposal_log_ratios == pytest.approx({
        "mild": math.log(0.3 / 0.5625),
        "strong": math.log(0.3 / 0.8372093023255814),
    })
    assert sampler.audit_proposal_component_log_ratios["mild"] == pytest.approx({
        "sensor.outcome.window": math.log(0.3 / 0.5625),
    })


def test_categorical_cross_evaluation_requires_every_declared_proposal():
    sampler = ImportanceSampler(seed=1, audit_proposal_ids=("mild", "strong"))

    with pytest.raises(SimulationConfigurationError, match="Missing categorical"):
        sampler.categorical(
            [0.7, 0.3], [0.7, 0.3], audit_probs_bias={"mild": [0.5, 0.5]}
        )


def test_continuous_draws_cross_evaluate_nominal_and_biased_proposals():
    sampler = ImportanceSampler(
        seed=19, audit_proposal_ids=("nominal", "biased")
    )

    sampler.normal(
        mu_base=0.0, sigma_base=1.0,
        mu_bias=-0.5, sigma_bias=0.6,
        component="fault.trigger",
        audit_params_bias={
            "nominal": (0.0, 1.0),
            "biased": (-0.5, 0.6),
        },
    )
    sampler.shifted_lognormal(
        shift_base=50.0, mu_base=3.0, sigma_base=0.5,
        shift_bias=50.0, mu_bias=3.4, sigma_bias=0.7,
        component="network.delay.window",
        audit_params_bias={
            "nominal": (50.0, 3.0, 0.5),
            "biased": (50.0, 3.4, 0.7),
        },
    )

    assert sampler.audit_proposal_log_ratios["nominal"] == pytest.approx(0.0)
    assert sampler.audit_proposal_log_ratios["biased"] == pytest.approx(
        sampler.component_log_ratio
    )
    assert set(sampler.audit_proposal_component_log_ratios["biased"]) == {
        "fault.trigger", "network.delay.window",
    }


def test_sensor_cross_evaluation_respects_each_proposals_own_window(monkeypatch):
    sampler = ImportanceSampler(
        seed=31, audit_proposal_ids=("nominal", "early", "late")
    )
    sensor_cfg = SimpleNamespace(
        position=(0.0, 0.0, 0.0), orientation=0.0, fov_degrees=120,
        radius=20.0, probA_det=0.7, probA_wrong=0.0, probA_miss=0.3,
        probA_ghost=0.0, probB_det=0.7, probB_wrong=0.0, probB_miss=0.3,
        probB_ghost=0.0, unc_pos=0.0, unc_vel=0.0, unc_type="normal",
    )
    sensor = SectorLogic(
        sensor_cfg,
        sensor_bias={
            "miss_scale": 12.0,
            "active_window": {"start_ms": 100, "end_ms": 200},
            "audit_proposals": {
                "nominal": {"miss_scale": 1.0},
                "early": {
                    "miss_scale": 12.0,
                    "active_window": {"start_ms": 100, "end_ms": 200},
                },
                "late": {
                    "miss_scale": 8.0,
                    "active_window": {"start_ms": 300, "end_ms": 400},
                },
            },
        },
        sampler=sampler,
    )

    class AlwaysInside:
        covers = staticmethod(lambda _shape: True)
        intersects = staticmethod(lambda _shape: False)

    sensor.fov_polygon_A = AlwaysInside()
    sensor.fov_polygon_B = AlwaysInside()
    monkeypatch.setattr(sensor, "compute_result_A", lambda poll, _target, _shape: poll)
    monkeypatch.setattr(sampler.rng, "random", lambda: 0.99)
    target = SimpleNamespace(id=17)
    sensor.logic_step(target, [], sim_time=150, shape_by_actor_id={17: object()})

    assert sampler.audit_proposal_log_ratios["nominal"] == pytest.approx(0.0)
    assert sampler.audit_proposal_log_ratios["early"] == pytest.approx(
        sampler.component_log_ratio
    )
    assert sampler.audit_proposal_log_ratios["late"] == pytest.approx(0.0)


def test_network_and_fault_cross_evaluation_accumulate_full_proposal_ratios():
    sampler = ImportanceSampler(
        seed=37, audit_proposal_ids=("nominal", "delivery", "fault")
    )
    network = NetworkLogicLogNorm(
        {
            "packet_drop_rate": 0.01,
            "delay_min": 50.0,
            "delay_avg": 100.0,
            "jitter": 20.0,
        },
        sampler=sampler,
        bias_cfg={
            "packet_drop_rate": 0.01,
            "delay_min": 50.0,
            "delay_avg": 200.0,
            "jitter": 20.0,
            "active_window": {"start_ms": 100, "end_ms": 200},
            "audit_proposals": {
                "nominal": {},
                "delivery": {
                    "packet_drop_rate": 0.01,
                    "delay_min": 50.0,
                    "delay_avg": 200.0,
                    "jitter": 20.0,
                    "active_window": {"start_ms": 100, "end_ms": 200},
                },
                "fault": {},
            },
        },
    )
    SITLogic(
        0.05,
        {"offline": 1500, "trigger_mean": 3000.0, "trigger_var": 1000.0},
        sampler=sampler,
        bias_cfg={
            "trigger_mean_add": 0.0,
            "trigger_var_scale": 1.0,
            "audit_proposals": {
                "nominal": {},
                "delivery": {},
                "fault": {"trigger_mean_add": -500.0, "trigger_var_scale": 0.5},
            },
        },
    )
    network.compute_De2e(sim_time=150)
    network.should_drop_packet(sim_time=150)

    assert sampler.audit_proposal_log_ratios["nominal"] == pytest.approx(0.0)
    assert sampler.audit_proposal_component_log_ratios["delivery"][
        "fault.trigger"
    ] == pytest.approx(0.0)
    assert sampler.audit_proposal_component_log_ratios["fault"][
        "network.delay.window"
    ] == pytest.approx(0.0)
    assert sampler.audit_proposal_log_ratios["delivery"] == pytest.approx(
        sampler.component_log_ratios["network.delay.window"]
    )


def test_proposal_must_cover_nominal_support():
    sampler = ImportanceSampler(seed=0)

    with pytest.raises(ValueError, match="excludes"):
        sampler.bernoulli(p_base=0.1, p_bias=0.0)
    with pytest.raises(ValueError, match="excludes"):
        sampler.categorical([0.5, 0.5], [1.0, 0.0])


def test_impossible_nominal_outcome_gets_zero_weight():
    sampler = ImportanceSampler(seed=1)

    assert sampler.bernoulli(p_base=0.0, p_bias=0.5) is True
    assert sampler.log_weight == -math.inf
    assert sampler.weight() == 0.0


def test_categorical_zero_rng_draw_skips_zero_probability_entries():
    class ZeroRng:
        @staticmethod
        def random():
            return 0.0

    assert _sample_categorical(ZeroRng(), [0.0, 0.0, 1.0]) == 2


def test_categorical_zero_mass_boundary_keeps_log_weight_finite(monkeypatch):
    sampler = ImportanceSampler(seed=1)
    monkeypatch.setattr(sampler.rng, "random", lambda: 0.0)

    outcome = sampler.categorical(
        probs_base=[0.0, 0.25, 0.75],
        probs_bias=[0.0, 0.50, 0.50],
    )

    assert outcome == 1
    assert sampler.log_weight == pytest.approx(math.log(0.25 / 0.50))
    assert math.isfinite(sampler.log_weight)


def test_categorical_numerical_guard_returns_last_positive_entry():
    class UpperBoundaryRng:
        @staticmethod
        def random():
            return 1.0

    assert _sample_categorical(UpperBoundaryRng(), [0.5, 0.5, 0.0]) == 1


def test_shifted_lognormal_evaluates_nominal_density_at_total_value():
    seed = 17
    shift_base = 10.0
    shift_bias = 20.0
    mu_base, sigma_base = 1.0, 0.5
    mu_bias, sigma_bias = 1.3, 0.7
    expected_variable = random.Random(seed).lognormvariate(mu_bias, sigma_bias)
    expected_total = shift_bias + expected_variable
    expected_log_weight = (
        _logpdf_lognormal(expected_total - shift_base, mu_base, sigma_base)
        - _logpdf_lognormal(expected_variable, mu_bias, sigma_bias)
    )
    sampler = ImportanceSampler(seed=seed)

    total = sampler.shifted_lognormal(
        shift_base,
        mu_base,
        sigma_base,
        shift_bias,
        mu_bias,
        sigma_bias,
    )

    assert total == pytest.approx(expected_total)
    assert sampler.log_weight == pytest.approx(expected_log_weight)


def test_bias_configuration_requires_sampler_even_when_parameters_are_identity():
    require_sampler_for_bias(None, None, "test")
    require_sampler_for_bias(None, {}, "test")
    require_sampler_for_bias(object(), {"scale": 1.0}, "test")
    with pytest.raises(ValueError, match="sampler-less execution must remain nominal"):
        require_sampler_for_bias(None, {"scale": 1.0}, "test")


def test_network_models_reject_bias_without_sampler_and_default_to_base():
    lognormal_cfg = {
        "packet_drop_rate": 0.01,
        "delay_min": 50.0,
        "delay_avg": 100.0,
        "jitter": 100.0,
    }
    with pytest.raises(ValueError, match="requires an ImportanceSampler"):
        NetworkLogicLogNorm(
            lognormal_cfg,
            bias_cfg={**lognormal_cfg, "packet_drop_rate": 0.2},
        )
    nominal_lognormal = NetworkLogicLogNorm(lognormal_cfg)
    assert nominal_lognormal.drop_rate_bias == nominal_lognormal.drop_rate
    assert nominal_lognormal.delay_avg_bias == nominal_lognormal.delay_avg

    uniform_cfg = {"packet_drop_rate": 0.01, "delay": 50.0, "jitter": 100.0}
    with pytest.raises(ValueError, match="requires an ImportanceSampler"):
        NetworkLogicUniform(
            uniform_cfg,
            bias_cfg={**uniform_cfg, "packet_drop_rate": 0.2},
        )
    nominal_uniform = NetworkLogicUniform(uniform_cfg)
    assert nominal_uniform.drop_rate_bias == nominal_uniform.drop_rate
    assert nominal_uniform.delay_bias == nominal_uniform.delay


def test_samplerless_fault_uses_base_parameters_and_rejects_bias(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "source.simulation_environment.communication.fault_models.sit.random.normalvariate",
        lambda mean, sigma: calls.append((mean, sigma)) or mean,
    )
    cfg = {"offline": 200, "trigger_mean": 100.0, "trigger_var": 20.0}
    nominal = SITLogic(0.05, cfg)
    assert nominal.trigger_step == 100
    assert calls == [(100.0, 20.0)]

    with pytest.raises(ValueError, match="requires an ImportanceSampler"):
        SITLogic(
            0.05,
            cfg,
            bias_cfg={"trigger_mean_add": -50.0, "trigger_var_scale": 0.5},
        )
