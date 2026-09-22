#!/usr/bin/env python
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

from enum import IntEnum
import carla

class BaseScenario:

    class State(IntEnum):
        END_DONE = 0
        END_COLLISION = 1
        IN_PROGRESS = 2

    class ErrorCode(IntEnum):
        ERROR_LOADING_CFG = 100
        OK = 200

    def __init__(self, experiment_name: str, world: carla.World, carla_map: carla.Map, cfg: dict, delta_time):
        self.world = world
        self.carla_map = carla_map
        self.cfg = cfg
        self.experiment_name = experiment_name
        self.delta_time = delta_time

    def generate_scenario(self) -> ErrorCode:
        pass

    def pre_run(self):
        pass

    def is_scenario_done(self, sim_time) -> State:
       pass

    def run_step(self, sim_time, extra) -> State:
        pass

    @staticmethod
    def get_scenario_map() -> str:
        return ""

    @staticmethod
    def state_to_string(state:State) -> str:
        if state == BaseScenario.State.END_DONE:
            return "Done"
        elif state == BaseScenario.State.END_COLLISION:
            return "Collision"
        elif state == BaseScenario.State.IN_PROGRESS:
            return "In progress"

        return "Unknown"

    def get_ego_agent(self):
        pass

    def get_adv_agent(self):
        pass

    @staticmethod
    def destroy_actors(*actors) -> None:
        """Best-effort cleanup for fully or partially constructed scenarios."""
        for actor in actors:
            if actor is None:
                continue
            try:
                actor.destroy()
            except RuntimeError:
                # Reloaded worlds may have already invalidated an actor handle.
                pass
