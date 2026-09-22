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
"""Pin the single-shot outage state machine and its trigger-timing law.

`SITLogic` takes the link down once, holds it down for a configured duration,
and then stays up for the rest of the episode. The exact tick boundaries matter:
they decide how long the ego runs without sensor messages, which is what the
rare-event study is measuring. Two boundaries are easy to get wrong and are
pinned explicitly below -- the trigger tick still reports ONLINE, and the outage
ends strictly *after* the configured duration.
"""

import random

import pytest

from source.simulation_environment.communication.fault_models import fault_models_factory
from source.simulation_environment.communication.fault_models.fault_logic_base import ComEvent
from source.simulation_environment.communication.fault_models.sit import SITLogic
from source.simulation_environment.configuration_error import SimulationConfigurationError


DELTA_TIME = 0.05
NEVER = 10 ** 9


@pytest.fixture(autouse=True)
def _isolate_global_rng():
    """The stochastic-trigger tests seed the process-wide generator; hand it back
    untouched so the suite stays independent of file ordering."""
    state = random.getstate()
    yield
    random.setstate(state)


def _sit(offline=100, **overrides):
    cfg = {"offline": offline, "trigger": NEVER}
    cfg.update(overrides)
    return SITLogic(DELTA_TIME, cfg)


class RecordingSampler:
    """Captures the moments each trigger draw is issued with."""

    def __init__(self, value=1234.6):
        self.calls = []
        self._value = value

    def normal(self, **kwargs):
        self.calls.append(kwargs)
        return self._value

    @property
    def last(self):
        return self.calls[-1]


class TestValidation:

    def test_a_deterministic_trigger_config_is_accepted(self):
        assert SITLogic.validate_cfg({"offline": 100, "trigger": 500}) == (True, "")

    def test_a_stochastic_trigger_config_is_accepted(self):
        assert SITLogic.validate_cfg(
            {"offline": 100, "trigger_mean": 1000, "trigger_var": 200}
        ) == (True, "")

    def test_the_offline_duration_is_required(self):
        valid, reason = SITLogic.validate_cfg({"trigger": 500})

        assert not valid
        assert "offline" in reason

    @pytest.mark.parametrize("variance", [0, -1, -0.5])
    def test_a_stochastic_trigger_needs_a_positive_variance(self, variance):
        valid, reason = SITLogic.validate_cfg({"offline": 100, "trigger_var": variance})

        assert not valid
        assert "trigger_var must be > 0" in reason

    @pytest.mark.parametrize("variance", [0, -1])
    def test_a_deterministic_trigger_ignores_the_variance(self, variance):
        """With an explicit `trigger` there is no distribution to parameterise."""
        assert SITLogic.validate_cfg(
            {"offline": 100, "trigger": 500, "trigger_var": variance}
        ) == (True, "")

    def test_an_unconfigured_trigger_is_accepted_and_never_fires(self):
        """Both trigger moments default to `sys.maxsize`, which is a valid --
        if pathological -- 'never' setting."""
        assert SITLogic.validate_cfg({"offline": 100}) == (True, "")


class TestTriggerTiming:

    def test_an_explicit_trigger_is_used_verbatim(self):
        assert _sit(trigger=777).trigger_step == 777

    def test_an_explicit_trigger_is_rounded_to_a_whole_tick(self):
        assert _sit(trigger=777.6).trigger_step == 778

    def test_an_explicit_trigger_bypasses_the_sampler(self):
        sampler = RecordingSampler()

        model = SITLogic(
            DELTA_TIME,
            {"offline": 100, "trigger": 777, "trigger_mean": 1000, "trigger_var": 200},
            sampler=sampler,
        )

        assert sampler.calls == [], "a deterministic trigger has no law to weight"
        assert model.trigger_step == 777

    def test_a_stochastic_trigger_draws_from_the_configured_moments(self):
        random.seed(31)
        expected = round(random.normalvariate(1000, 200))
        random.seed(31)

        model = SITLogic(DELTA_TIME, {"offline": 100, "trigger_mean": 1000, "trigger_var": 200})

        assert model.trigger_step == expected

    def test_a_stochastic_trigger_actually_varies(self):
        random.seed(5)

        steps = {
            SITLogic(DELTA_TIME, {"offline": 100, "trigger_mean": 1000, "trigger_var": 200}).trigger_step
            for _ in range(20)
        }

        assert len(steps) > 1

    def test_sampler_receives_the_base_and_proposal_moments(self):
        sampler = RecordingSampler()

        model = SITLogic(
            DELTA_TIME,
            {"offline": 100, "trigger_mean": 1000, "trigger_var": 200},
            sampler=sampler,
            bias_cfg={"trigger_mean": 700, "trigger_var": 50},
        )

        assert sampler.last == {
            "mu_base": 1000,
            "sigma_base": 200,
            "mu_bias": 700,
            "sigma_bias": 50,
            "component": "fault.trigger",
        }
        assert model.trigger_step == 1235

    def test_an_empty_bias_block_proposes_the_base_law(self):
        sampler = RecordingSampler()

        SITLogic(
            DELTA_TIME,
            {"offline": 100, "trigger_mean": 1000, "trigger_var": 200},
            sampler=sampler,
            bias_cfg={},
        )

        assert sampler.last["mu_bias"] == sampler.last["mu_base"]
        assert sampler.last["sigma_bias"] == sampler.last["sigma_base"]

    def test_deltas_shift_the_proposal_and_leave_the_base_law_alone(self):
        """`trigger_mean_add` is additive and `trigger_var_scale` multiplicative,
        applied on top of any explicit proposal moments. The base law is what the
        likelihood ratio divides by, so it must not move."""
        sampler = RecordingSampler()

        SITLogic(
            DELTA_TIME,
            {"offline": 100, "trigger_mean": 1000, "trigger_var": 200},
            sampler=sampler,
            bias_cfg={
                "trigger_mean": 800,
                "trigger_var": 50,
                "trigger_mean_add": -100.0,
                "trigger_var_scale": 2.0,
            },
        )

        assert sampler.last["mu_base"] == 1000
        assert sampler.last["sigma_base"] == 200
        assert sampler.last["mu_bias"] == 700.0
        assert sampler.last["sigma_bias"] == 100.0

    def test_deltas_apply_to_the_base_moments_when_no_proposal_is_given(self):
        sampler = RecordingSampler()

        SITLogic(
            DELTA_TIME,
            {"offline": 100, "trigger_mean": 1000, "trigger_var": 200},
            sampler=sampler,
            bias_cfg={"trigger_mean_add": 250.0, "trigger_var_scale": 0.5},
        )

        assert sampler.last["mu_bias"] == 1250.0
        assert sampler.last["sigma_bias"] == 100.0

    def test_the_drawn_trigger_is_rounded_to_a_whole_tick(self):
        model = SITLogic(
            DELTA_TIME,
            {"offline": 100, "trigger_mean": 1000, "trigger_var": 200},
            sampler=RecordingSampler(value=500.4),
            bias_cfg={"trigger_mean": 500},
        )

        assert model.trigger_step == 500


class TestBiasGuards:

    def test_a_deterministic_trigger_override_is_refused(self):
        with pytest.raises(SimulationConfigurationError, match="fault_bias.trigger"):
            SITLogic(DELTA_TIME, {"offline": 100}, sampler=RecordingSampler(),
                     bias_cfg={"trigger": 100})

    def test_bias_without_a_sampler_is_refused(self):
        with pytest.raises(SimulationConfigurationError, match="Fault bias"):
            SITLogic(DELTA_TIME, {"offline": 100}, sampler=None,
                     bias_cfg={"trigger_mean": 100})

    @pytest.mark.parametrize("variance", [0, -1])
    def test_a_non_positive_base_variance_is_refused_under_a_sampler(self, variance):
        with pytest.raises(SimulationConfigurationError, match="positive base"):
            SITLogic(
                DELTA_TIME,
                {"offline": 100, "trigger_mean": 1000, "trigger_var": variance},
                sampler=RecordingSampler(),
                bias_cfg={"trigger_mean_add": 1.0},
            )

    @pytest.mark.parametrize("scale", [0.0, -1.0])
    def test_a_non_positive_proposal_variance_is_refused_under_a_sampler(self, scale):
        with pytest.raises(SimulationConfigurationError, match="positive base"):
            SITLogic(
                DELTA_TIME,
                {"offline": 100, "trigger_mean": 1000, "trigger_var": 200},
                sampler=RecordingSampler(),
                bias_cfg={"trigger_var_scale": scale},
            )


class TestOutageStateMachine:

    def test_the_link_starts_online(self):
        model = _sit(trigger=100)

        assert model.online is True
        assert model.step(0) is ComEvent.ONLINE

    def test_a_zero_offline_duration_never_takes_the_link_down(self):
        """`offline = 0` pre-arms `triggered`, which is how a config disables the
        fault model entirely."""
        model = _sit(offline=0, trigger=10)

        events = [model.step(t) for t in (0, 10, 20, 1000)]

        assert events == [ComEvent.ONLINE] * 4
        assert model.online is True

    def test_the_link_stays_up_before_the_trigger(self):
        model = _sit(offline=50, trigger=100)

        assert [model.step(t) for t in (0, 50, 99)] == [ComEvent.ONLINE] * 3
        assert model.online is True

    def test_the_trigger_tick_flips_the_state_but_still_reports_online(self):
        """A one-tick lag: `step` sets `online = False` and returns ONLINE on the
        same call. The outage is only *reported* from the following tick."""
        model = _sit(offline=50, trigger=100)
        model.step(99)

        event = model.step(100)

        assert event is ComEvent.ONLINE
        assert model.online is False
        assert model.start_offline_time == 100

    def test_the_outage_is_reported_from_the_tick_after_the_trigger(self):
        model = _sit(offline=50, trigger=100)
        model.step(100)

        assert model.step(101) is ComEvent.OFFLINE

    @pytest.mark.parametrize("elapsed", [1, 25, 49, 50])
    def test_the_link_stays_down_through_the_configured_duration(self, elapsed):
        """The recovery test is `diff > offline`, so the tick at exactly the
        configured duration is still an outage."""
        model = _sit(offline=50, trigger=100)
        model.step(100)

        assert model.step(100 + elapsed) is ComEvent.OFFLINE
        assert model.online is False

    def test_the_link_recovers_strictly_after_the_configured_duration(self):
        model = _sit(offline=50, trigger=100)
        model.step(100)
        model.step(150)

        assert model.step(151) is ComEvent.ONLINE
        assert model.online is True

    def test_a_late_first_tick_after_the_trigger_recovers_immediately(self):
        model = _sit(offline=50, trigger=100)
        model.step(100)

        assert model.step(5000) is ComEvent.ONLINE

    def test_recovery_is_single_shot(self):
        """`triggered` is never reset, so the link goes down at most once per
        episode however long the run continues."""
        model = _sit(offline=50, trigger=100)
        for tick in (0, 100, 151):
            model.step(tick)

        events = [model.step(t) for t in range(200, 5000, 250)]

        assert set(events) == {ComEvent.ONLINE}
        assert model.online is True

    def test_the_full_outage_window_has_the_expected_length(self):
        model = _sit(offline=50, trigger=100)

        offline_ticks = [t for t in range(0, 200) if model.step(t) is ComEvent.OFFLINE]

        assert offline_ticks == list(range(101, 151))

    def test_the_trigger_is_evaluated_against_a_monotonically_advancing_clock(self):
        """A tick that jumps past the trigger still fires it."""
        model = _sit(offline=50, trigger=100)

        assert model.step(0) is ComEvent.ONLINE
        model.step(4000)

        assert model.online is False
        assert model.start_offline_time == 4000


class TestFactory:

    def test_the_sit_type_yields_the_sit_model(self):
        model = fault_models_factory.get_fault_model(
            DELTA_TIME, {"type": "sit", "offline": 100, "trigger": 500}
        )

        assert isinstance(model, SITLogic)
        assert model.trigger_step == 500

    def test_an_unknown_type_yields_no_model(self):
        assert fault_models_factory.get_fault_model(DELTA_TIME, {"type": "gremlins"}) is None

    def test_sampler_and_bias_reach_the_constructed_model(self):
        sampler = RecordingSampler()

        model = fault_models_factory.get_fault_model(
            DELTA_TIME,
            {"type": "sit", "offline": 100, "trigger_mean": 1000, "trigger_var": 200},
            sampler=sampler,
            bias_cfg={"trigger_mean": 400},
        )

        assert model.sampler is sampler
        assert sampler.last["mu_bias"] == 400

    def test_validate_cfg_rejects_an_unknown_type(self):
        valid, reason = fault_models_factory.validate_cfg({"fault": {"type": "gremlins"}})

        assert not valid
        assert reason == "Unknown fault type"

    def test_validate_cfg_delegates_to_the_selected_model(self):
        assert fault_models_factory.validate_cfg(
            {"fault": {"type": "sit", "offline": 100, "trigger": 500}}
        ) == (True, "")

    def test_validate_cfg_surfaces_the_model_diagnostic(self):
        valid, reason = fault_models_factory.validate_cfg({"fault": {"type": "sit"}})

        assert not valid
        assert "offline" in reason


class TestRareEventConfigValidation:

    SIT_COMM = {"fault": {"type": "sit", "offline": 100, "trigger_mean": 1000, "trigger_var": 200}}

    def test_an_absent_bias_section_is_accepted(self):
        assert fault_models_factory.validate_rare_event_cfg({}, self.SIT_COMM) == (True, "")

    @pytest.mark.parametrize(
        "bias",
        [
            {"trigger_mean": 500},
            {"trigger_var": 50},
            {"trigger_mean_add": -100.0},
            {"trigger_var_scale": 2.0},
            {"trigger_mean": 500, "trigger_var": 50,
             "trigger_mean_add": 10.0, "trigger_var_scale": 1.5},
        ],
    )
    def test_supported_bias_fields_are_accepted(self, bias):
        assert fault_models_factory.validate_rare_event_cfg(
            {"fault_bias": bias}, self.SIT_COMM
        ) == (True, "")

    @pytest.mark.parametrize("field", ["trigger", "offline", "typo"])
    def test_unsupported_bias_fields_are_named_in_the_diagnostic(self, field):
        valid, reason = fault_models_factory.validate_rare_event_cfg(
            {"fault_bias": {field: 1}}, self.SIT_COMM
        )

        assert not valid
        assert field in reason

    @pytest.mark.parametrize("scale", [0, -1.0])
    def test_a_non_positive_variance_scale_is_rejected(self, scale):
        valid, reason = fault_models_factory.validate_rare_event_cfg(
            {"fault_bias": {"trigger_var_scale": scale}}, self.SIT_COMM
        )

        assert not valid
        assert "trigger_var_scale must be > 0" in reason

    @pytest.mark.parametrize("variance", [0, -1])
    def test_a_non_positive_proposal_variance_is_rejected(self, variance):
        valid, reason = fault_models_factory.validate_rare_event_cfg(
            {"fault_bias": {"trigger_var": variance}}, self.SIT_COMM
        )

        assert not valid
        assert "trigger_var must be > 0" in reason

    def test_bias_is_not_checked_without_a_communication_block(self):
        """Characterisation test: with no `communication` section to identify the
        fault model, unsupported fields pass unchecked."""
        assert fault_models_factory.validate_rare_event_cfg(
            {"fault_bias": {"nonsense": 1}}, None
        ) == (True, "")
