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
"""Pin the config-to-world-settings mapping.

`apply_sim_settings` decides whether the server renders. Rendering is what
exhausts after ~31 world loads and takes the server down mid-job, so the default
here is load-bearing for both crash rate and reproducibility: flipping it silently
would change every campaign's simulator provenance.
"""

from types import SimpleNamespace

import pytest

from source.simulation_environment.runner.runner import apply_sim_settings


def _settings():
    """Stand-in for carla.WorldSettings: a plain attribute bag is enough."""
    return SimpleNamespace(
        synchronous_mode=False,
        fixed_delta_seconds=0.0,
        substepping=False,
        no_rendering_mode=False,
    )


def _cfg(**sim):
    base = {"delta_time": 0.01}
    base.update(sim)
    return {"sim": base}


def test_applies_the_synchronous_physics_contract():
    settings = apply_sim_settings(_settings(), _cfg())

    assert settings.synchronous_mode is True
    assert settings.fixed_delta_seconds == 0.01
    assert settings.substepping is True


def test_rendering_stays_on_when_the_config_is_silent():
    """Existing campaign configs carry no such key and must not change behaviour."""
    settings = apply_sim_settings(_settings(), _cfg())

    assert settings.no_rendering_mode is False


@pytest.mark.parametrize("configured, expected", [
    (True, True),
    (False, False),
    (1, True),
    (0, False),
])
def test_no_rendering_mode_follows_the_config(configured, expected):
    settings = apply_sim_settings(_settings(), _cfg(no_rendering_mode=configured))

    assert settings.no_rendering_mode is expected


def test_returns_the_same_object_it_was_given():
    """`main` re-applies this object after every reload_world; it must not copy."""
    original = _settings()

    assert apply_sim_settings(original, _cfg(no_rendering_mode=True)) is original


def test_delta_time_is_taken_from_the_config_not_defaulted():
    settings = apply_sim_settings(_settings(), _cfg(delta_time=0.05))

    assert settings.fixed_delta_seconds == 0.05
