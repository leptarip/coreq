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

"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specified route
"""
import math
from collections import deque
from typing import Optional

import carla

from agents.navigation.basic_agent import BasicAgent
from carla import VehicleControl

import source.simulation_environment.agents.traj_shape as tjs

#from source.simulation_environment.cython.basic_agent_cy import BasicAgent

class DynamicAgent(BasicAgent):
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    def __init__(self,
                 vehicle: carla.Actor,
                 opt_dict=None,
                 map_inst=None,
                 grp_inst=None):
        """
        Initialization the agent parameters, the local and the global planner.
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.

        """
        if opt_dict is None:
            opt_dict = dict()
        opt_dict["use_bbs_detection"] = False #disable BasicAgent reactionary behavior
        opt_dict['sampling_resolution'] = 0.1 #distance between two sampled waypoint, used by global planner

        super().__init__(vehicle, target_speed=60, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self.route_polygon: Optional[tjs.RoutePolygon] = None
        self.period = 5 # make the step every period milliseconds

    def get_actor(self):
        return self._vehicle

    def run_step(self, list_obs=None) -> VehicleControl:
        """Execute one step of navigation and return the control."""
        return self._local_planner.run_step()

    def step(self, sim_time, list_obs=None) -> None:
        """Execute one step of navigation and move ego."""
        if sim_time % self.period == 0:
            control = self.run_step(list_obs)
            self._vehicle.apply_control(control)

    def get_path_plan(self):
        a = list(zip(*self._local_planner.get_plan()))
        if len(a) < 1:
            return []
        return a[0]

    def get_path_polygon(self) -> tjs.RoutePolygon:
        if self.route_polygon is None:
            self.route_polygon = tjs.get_route_polygon(self._vehicle, self.get_plan())
        return self.route_polygon

    def get_plan(self) -> deque:
        return self._local_planner.get_plan()

    def show_path(self):
        tjs.show_path(self.get_plan())

    def destroy(self):
        carla.command.DestroyActor(self._vehicle)

    def manual_way_point_update(self, list_obs) -> None:
        """
        While moving without using the local planner, update the waypoint list
        """
        # Add more waypoints too few in the horizon
        if not self._local_planner._stop_waypoint_creation and len(
                self._local_planner._waypoints_queue) < self._local_planner._min_waypoint_queue_length:
            self._local_planner._compute_next_waypoints(k=self._local_planner._min_waypoint_queue_length)

        # Purge the queue of obsolete waypoints
        veh_location = self._vehicle.get_location()
        vel = self._vehicle.get_velocity()
        vehicle_speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        self._local_planner._min_distance = self._local_planner._base_min_distance + self._local_planner._distance_ratio * vehicle_speed

        num_waypoint_removed = 0
        for waypoint, _ in self._local_planner._waypoints_queue:

            if len(self._local_planner._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1  # Don't remove the last waypoint until very close by
            else:
                min_distance = self._local_planner._min_distance

            if veh_location.distance(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            else:
                break

        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                self._local_planner._waypoints_queue.popleft()

