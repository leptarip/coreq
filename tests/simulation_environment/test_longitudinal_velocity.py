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
"""Stopping-distance constraints with fixed response times and braking strength."""
import pytest

from source.simulation_environment.ego.LongVelConstrainer import LongVelController


def controller(response_time):
    return LongVelController(
        a_max_acc=2.0, a_min_deceleration=4.0, a_max_deceleration=8.0,
        min_response_time=response_time, max_response_time=response_time,
    )


def test_zero_response_time_matches_braking_distance():
    model = controller(0.0)
    # At 8 m/s with 4 m/s^2 braking, stopping takes 8 m, plus the 0.8 m margin.
    velocity, unconstrained, feasible = model.compute_velocity(8.8)
    assert feasible
    assert velocity == pytest.approx(8.0)
    assert unconstrained == pytest.approx(8.0)
    assert model.get_min_distance(8.0) == pytest.approx(8.0)


def test_response_delay_includes_acceleration_before_braking():
    model = controller(0.5)
    # Starting at 4 m/s: 2.25 m during the delay, then 3.125 m braking from 5 m/s.
    assert model.get_min_distance(4.0) == pytest.approx(5.375)
    velocity, unconstrained, feasible = model.compute_velocity(6.175)
    assert feasible
    assert velocity == pytest.approx(4.0)
    assert unconstrained == pytest.approx(4.0)


def test_obstacle_inside_margin_is_infeasible_and_clamps_velocity_to_zero():
    velocity, unconstrained, feasible = controller(0.0).compute_velocity(0.4)
    assert not feasible
    assert velocity == 0.0
    assert unconstrained < 0.0
