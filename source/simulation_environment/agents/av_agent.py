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

import math
from collections import deque
from typing import Optional

import carla

from agents.navigation.basic_agent import BasicAgent
# from source.simulation_environment.cython.basic_agent_cy import BasicAgent
from carla import VehicleControl

import source.simulation_environment.agents.traj_shape as tjs
from source.simulation_environment.agents.obs_detection_2d_gt import ObsDetection2D
from source.simulation_environment.agents.perception_2d_gt import Perception2Dgt
from source.simulation_environment.agents.traj_shape import RoutePolygon
from source.simulation_environment.cfg.vehicle_cfg import VehicleCfg
from source.simulation_environment.ego.LongVelConstrainer import LongVelController
from source.simulation_environment.mock.mock import MockSensor
import source.simulation_environment.logs.ego_logs as ego_logs
import source.simulation_environment.logs.redis_data as rd




class AVAgent(BasicAgent):
    """
    AVAgent implements an agent that navigates the scene and as 2D perception and longitudinal object avoidance.
    """

    def __init__(self,
                 vehicle: carla.Actor,
                 cfg: VehicleCfg,
                 experiment: rd.SimRunKeys,
                 map_buildings,
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
        opt_dict["use_bbs_detection"] = False  # disable BasicAgent reactionary behavior
        opt_dict['sampling_resolution'] = 0.2  # distance between two sampled waypoint, used by global planner
        opt_dict['max_throttle'] = cfg.max_throttle
        opt_dict['max_brake'] = cfg.max_break
        opt_dict['distance_ratio'] = 0.1
        opt_dict['base_min_distance'] = 1.5
        opt_dict['target_speed'] = cfg.target_speed

        super().__init__(vehicle, cfg.target_speed, opt_dict, map_inst, grp_inst)
        self.name = cfg.role_name
        self.ego = vehicle
        self.target = None

        self.mock_sensor = MockSensor(pos_x=self.ego.get_location().x,
                                      pos_y=self.ego.get_location().y,
                                      yaw=self.ego.get_transform().rotation.yaw,
                                      roll=self.ego.get_transform().rotation.roll,
                                      pitch=self.ego.get_transform().rotation.pitch)

        self.perception = Perception2Dgt(map_buildings,
                                         self.mock_sensor,
                                         cfg.sensor_fov,
                                         cfg.sensor_range,
                                         cfg.sensor_resolution)

        self.detection = ObsDetection2D(self.ego, cfg.prediction_window)

        self.lv_ctl = LongVelController(min_response_time=0.4,
                                        max_response_time=0.4,
                                        a_max_acc=cfg.max_acceleration,
                                        a_min_deceleration=cfg.max_deceleration)

        self.route_polygon: Optional[RoutePolygon] = None
        self._route_cache = tjs.RoutePolygonCache()
        self.reference_speed = cfg.target_speed / 3.6
        self.ad_period = cfg.decision_period
        self.v_selected = 0
        self.dist_to_obj = 0
        self.v_ctl = 0
        self.experiment = experiment

    def _update_sensor(self):
        trans = self.ego.get_transform()
        self.mock_sensor.location.x = trans.location.x
        self.mock_sensor.location.y = trans.location.y
        self.mock_sensor.rotation.yaw = trans.rotation.yaw
        self.mock_sensor.rotation.pitch = trans.rotation.pitch
        self.mock_sensor.rotation.roll = trans.rotation.roll

    def get_actor(self):
        return self._vehicle

    def set_target_to_observe(self, target):
        self.target = target

    def set_destination(self, end_location, start_location=None):
        super().set_destination(end_location, start_location)
        self.route_polygon = tjs.get_route_polygon(self._vehicle, self._local_planner.get_plan(),
                                                   cache=self._route_cache)

    def run_step(self, list_obs=None) -> VehicleControl:
        """Execute one step of navigation and return the control."""
        self.route_polygon = tjs.get_route_polygon(self._vehicle, self._local_planner.get_plan(),
                                                   cache=self._route_cache)

        # update the perception sensor according to new vehicle position
        self._update_sensor()
        # pass all relevant ground truth carla actors to perception
        self.perception.set_obstacles(list_obs)
        self.perception.step()

        # detect all obstacles intersecting the vehicle's path
        self.dist_to_obj = self.detection.step(self.perception.fov_polygon, self.route_polygon,
                                               self.perception.actors_map, self.target)
        self.v_ctl, _, _ = self.lv_ctl.compute_velocity(self.dist_to_obj)

        # check if there are detected obstacles within fov
        # if yes then apply the constraining on velocity
        # if not then apply the target speed - no restriction needed
        if self.detection.obs_detected is True:
            self.v_selected = min(self.reference_speed, self.v_ctl)
        else:
            self.v_selected = self.reference_speed

        # set the new reference speed to be followed by low level controlled (PID)
        self.set_target_speed(self.v_selected * 3.6)
        return self._local_planner.run_step()

    def step(self, sim_time, list_obs=None) -> None:
        """Execute one step of navigation and move ego."""
        if sim_time % self.ad_period == 0:
            control = self.run_step(list_obs)
            self._vehicle.apply_control(control)

    def get_path_plan(self):
        a = list(zip(*self._local_planner.get_plan()))
        if len(a) < 1:
            return []
        return a[0]

    def get_route_polygon(self) -> tjs.RoutePolygon:
        return self.route_polygon

    def get_plan(self) -> deque:
        return self._local_planner.get_plan()

    def destroy(self):
        carla.command.DestroyActor(self._vehicle)

    def manual_way_point_update(self, list_obs) -> None:
        """
        While moving without using the local planner, update the waypoint list
        """
        self.route_polygon = tjs.get_route_polygon(self._vehicle,
                                                   self._local_planner.get_plan(),
                                                   cache=self._route_cache)
        # update the perception sensor according to new vehicle position
        self._update_sensor()
        # pass all relevant ground truth carla actors to perception
        self.perception.set_obstacles(list_obs)
        self.perception.step()
        # detect all obstacles intersecting the vehicle's path
        _ = self.detection.step(self.perception.fov_polygon, self.route_polygon, self.perception.actors_map,
                                self.target)
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

    def log(self, sim_time):
        data_item = ego_logs.log_ego_data(sim_time,
                                          self._vehicle.get_acceleration(),
                                          self._vehicle.get_velocity(),
                                          self._vehicle.get_transform(),
                                          self.dist_to_obj,
                                          self.v_selected,
                                          self.v_ctl
                                          )

        rd.save_stream_ego_data(self.experiment.stream_data_name, data_item)
