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
from source.simulation_environment.scenario.scenario1.scenario_1 import ScenarioIntersection1, SCENARIO_1_NAME
from source.simulation_environment.scenario.scenario2.scenario_2 import ScenarioIntersection2, SCENARIO_2_NAME


def get_scenario(cfg: dict, experiment_name: str, world, carla_map, delta_time, sampler=None, seed=None):
    if cfg["scenario"] == SCENARIO_1_NAME:
        return ScenarioIntersection1(experiment_name, world, carla_map, cfg, delta_time, sampler=sampler, seed=seed)

    elif cfg["scenario"] == SCENARIO_2_NAME:
        return ScenarioIntersection2(experiment_name, world, carla_map, cfg, delta_time, sampler=sampler, seed=seed)

    else:
        return None


def get_scenario_map(cfg: dict):
    if cfg["scenario"] == SCENARIO_1_NAME:
        return ScenarioIntersection1.get_scenario_map()

    elif cfg["scenario"] == SCENARIO_2_NAME:
        return ScenarioIntersection2.get_scenario_map()

    else:
        return None
