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
"""Physical behavior of velocity-based acceleration estimation."""
import numpy as np
import pytest

from source.simulation_environment.agents.kalman_estimate import KalmanFilter


def test_stationary_observations_preserve_zero_motion():
    estimator = KalmanFilter(0.001)
    for elapsed_ms in (283, 317, 660):
        assert estimator.predict_acc(elapsed_ms, 0.0) == 0.0
    np.testing.assert_array_equal(estimator.get_state(), np.zeros((3, 1)))


@pytest.mark.parametrize("time_factor,elapsed", [(1.0, 0.1), (0.001, 100.0)])
def test_constant_acceleration_is_recovered_in_seconds_and_milliseconds(time_factor, elapsed):
    estimator = KalmanFilter(time_factor)
    acceleration = 2.0
    for step in range(1, 101):
        velocity = acceleration * step * 0.1
        estimate = estimator.predict_acc(elapsed, velocity)
    assert estimate == pytest.approx(acceleration, abs=0.01)
    assert estimator.get_state()[1, 0] == pytest.approx(20.0, abs=0.01)
