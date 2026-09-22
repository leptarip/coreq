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

import pytest

from source.design_exploration.simulator_interface.worker_protocol import (
    CONFIGURATION_ERROR_REASON,
    failure_reason_for_exception,
    is_retryable_failure,
    raise_worker_failure,
    should_retry_failure,
)
from source.simulation_environment.configuration_error import (
    SimulationConfigurationError,
    require_valid_configuration,
)
from source.simulation_environment.cfg.sensor_cfg import sample_uncertainty
from source.simulation_environment.communication.fault_models.sit import SITLogic
from source.simulation_environment.rare_events.wrapper import require_sampler_for_bias


def test_validation_failure_preserves_the_validator_diagnostic():
    reason = "rare_event.defensive_mixture_probability must be in [0, 1]"

    with pytest.raises(SimulationConfigurationError, match="defensive_mixture_probability") as caught:
        require_valid_configuration(False, reason, scope="rare-event")

    assert str(caught.value) == f"Invalid rare-event configuration: {reason}"


def test_configuration_failures_are_terminal_but_transient_failures_may_retry():
    config_error = SimulationConfigurationError("invalid proposal")
    assert failure_reason_for_exception(config_error) == CONFIGURATION_ERROR_REASON
    assert not is_retryable_failure(CONFIGURATION_ERROR_REASON)
    assert not should_retry_failure(
        CONFIGURATION_ERROR_REASON,
        attempts=1,
        max_retries=10,
    )
    assert is_retryable_failure("carla_dead")
    assert is_retryable_failure("exception")
    assert should_retry_failure("carla_dead", attempts=1, max_retries=10)
    assert not should_retry_failure("carla_dead", attempts=11, max_retries=10)


def test_worker_configuration_failure_is_restored_as_a_typed_error():
    with pytest.raises(SimulationConfigurationError, match="defensive_mixture_probability"):
        raise_worker_failure(
            reason=CONFIGURATION_ERROR_REASON,
            error="Invalid simulator configuration: defensive_mixture_probability",
            job_id="job-1",
        )


def test_samplerless_bias_guard_is_a_terminal_configuration_error():
    with pytest.raises(SimulationConfigurationError, match="Sensor bias configuration") as caught:
        require_sampler_for_bias(None, {"miss_scale": 2.0}, "Sensor")

    assert failure_reason_for_exception(caught.value) == CONFIGURATION_ERROR_REASON


def test_runtime_bias_safety_guards_are_typed_configuration_errors():
    with pytest.raises(SimulationConfigurationError) as uncertainty_error:
        sample_uncertainty(0.5, 0.1, "worst_case")
    assert failure_reason_for_exception(uncertainty_error.value) == CONFIGURATION_ERROR_REASON

    with pytest.raises(SimulationConfigurationError, match="fault_bias.trigger") as trigger_error:
        SITLogic(
            0.05,
            {"offline": 100},
            sampler=object(),
            bias_cfg={"trigger": 100},
        )
    assert failure_reason_for_exception(trigger_error.value) == CONFIGURATION_ERROR_REASON

    with pytest.raises(SimulationConfigurationError, match="positive base.*trigger_var") as variance_error:
        SITLogic(
            0.05,
            {"offline": 100, "trigger_mean": 100.0, "trigger_var": 0.0},
            sampler=object(),
            bias_cfg={"trigger_mean_add": 1.0},
        )
    assert failure_reason_for_exception(variance_error.value) == CONFIGURATION_ERROR_REASON
