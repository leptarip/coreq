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
import carla
from source.simulation_environment.cython.local_planner_cy import RoadOption

import source.simulation_environment.vehicle_constants as constants


class AdversaryActor:
    def __init__(self, world, carla_map, cfg: dict):
        self.world = world
        self.map = carla_map
        self.vehicle_bp = None
        self.initial_v = 0 #cfg["initial_v"]
        self.role_name = cfg["role_name"]
        self.target_v = cfg["target_v"]
        self.init_position_x = 35
        self.ini_position_y = -3.5
        self.destination_x = -30
        self.destination_y = -3.5

    def get_spawn_transform(self):
        transform = carla.Transform()
        transform.location.z = 0.10
        transform.location.x = self.init_position_x
        transform.location.y = self.ini_position_y
        transform.rotation.yaw = 180
        wp = self.map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        transform = wp.transform
        transform.location.z = 0.10
        return transform

    def get_actor_bp(self):
        self.vehicle_bp = self.world.get_blueprint_library().find(self.get_actor_bp_string())
        self.vehicle_bp.set_attribute('role_name', self.get_role_name())
        self.vehicle_bp.set_attribute('color', '255,0,0')
        return self.vehicle_bp

    @staticmethod
    def get_actor_bp_string():
        return 'vehicle.ford.mustang'

    def get_target_speed(self):
        """
        :return: speed in km/h
        """
        return self.target_v

    def get_initial_speed(self):
        """
        :return: speed in m/s
        """
        return self.initial_v / constants.MITER_KM_RATIO

    @staticmethod
    def get_control_max_throttle() -> float:
        """
        To be used with carla PID controller to limit the max acceleration
        :return:
        """
        return 0.9

    @staticmethod
    def get_control_max_break() -> float:
        """
        To be used with carla PID controller to limit the max deceleration
        :return:
        """
        return 0.5


    @staticmethod
    def get_max_acceleration() -> float:
        """
        need to find it out with experimentation and by tweaking your implemented controller
        :return:
        """
        return 2.0

    @staticmethod
    def get_max_deceleration() -> float:
        """
        need to find it out with experimentation nad by tweaking your implemented controller
        :return:
        """
        return 2.5

    def get_destination(self):
        location = carla.Location()
        location.x = self.destination_x
        location.y = self.destination_y
        wp = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        return wp.transform.location

    def get_role_name(self):
        return self.role_name


class EgoActor:

    def __init__(self, world: carla.World, carla_map, cfg: dict):
        self.world = world
        self.map = carla_map
        self.ego_bp = None
        self.position_x = 3.5
        self.position_y = 20
        self.destination_x = 3.2
        self.destination_y = -6.2
        self.initial_v = 0#cfg["initial_v"]
        self.role_name = cfg["role_name"]
        self.target_v = cfg["target_v"]

    def get_spawn_transform(self):
        transform = carla.Transform()
        transform.location.z = 0.10
        transform.location.x = self.position_x
        transform.location.y = self.position_y
        transform.rotation.yaw = -90
        wp = self.map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        transform = wp.transform
        transform.location.z = 0.10
        return transform

    def get_role_name(self):
        return self.role_name

    def get_actor_bp(self):
        self.ego_bp = self.world.get_blueprint_library().find(self.get_actor_bp_string())
        self.ego_bp.set_attribute('role_name', self.get_role_name())
        self.ego_bp.set_attribute('color', '0,0,255')
        return self.ego_bp

    @staticmethod
    def get_actor_bp_string():
        return 'vehicle.seat.leon'

    def get_target_speed(self):
        """
        :return: speed in km/h
        """
        return self.target_v

    def get_initial_speed(self):
        """
        :return: speed in m/s
        """
        return self.initial_v / constants.MITER_KM_RATIO

    @staticmethod
    def get_control_max_throttle() -> float:
        """
        To be used with carla PID controller to limit the max acceleration
        :return:
        """
        return 0.9

    @staticmethod
    def get_control_max_break() -> float:
        """
        To be used with carla PID controller to limit the max deceleration
        :return:
        """
        return 0.03

    @staticmethod
    def get_max_acceleration() -> float:
        """
        need to find it out with experimentation nad by tweaking your implemented controller
        :return:
        """
        return 4.0

    @staticmethod
    def get_max_deceleration() -> float:
        """
        need to find it out with experimentation nad by tweaking your implemented controller
        :return:
        """
        return 2.16 #3.5

    def get_destination(self):
        location = carla.Location()
        location.x = self.destination_x
        location.y = self.destination_y
        wp = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        return wp.transform.location

    def get_global_plan(self):
        """
        :return:  plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
        """
        t = self.get_spawn_transform()
        wp = self.map.get_waypoint(t.location,
                                   project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        yaw = wp.transform.rotation.yaw
        plan = [(wp, RoadOption.STRAIGHT)]

        wp = self.map.get_waypoint(self.get_destination(),
                                   project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        wp.transform.rotation.yaw = yaw
        plan.append((wp, RoadOption.STRAIGHT))

        return plan


class StaticActor3:

    def __init__(self, world, carla_map):
        self.world = world
        self.map = carla_map
        self.vehicle_bp = None

    def get_spawn_transform(self):
        transform = carla.Transform()
        transform.location.z = 0.25
        transform.location.x = -3.55464
        transform.location.y = 3
        wp = self.map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        transform = wp.transform
        transform.location.z = 0.10
        return transform

    def get_actor_bp(self):
        self.vehicle_bp = self.world.get_blueprint_library().find('vehicle.carlamotors.carlacola')
        return self.vehicle_bp


class StaticActor1:

    def __init__(self, world, carla_map):
        self.world = world
        self.map = carla_map
        self.vehicle_bp = None

    def get_spawn_transform(self):
        transform = carla.Transform()
        transform.location.z = 0.25
        transform.location.x = 19
        transform.location.y = 3.5
        transform.rotation.yaw = 0
        wp = self.map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        transform = wp.transform
        transform.location.z = 0.10
        return transform

    def get_actor_bp(self):
        self.vehicle_bp = self.world.get_blueprint_library().find("vehicle.ford.ambulance")
        self.vehicle_bp.set_attribute('color', '255,255,255')
        return self.vehicle_bp


class StaticActor2:

    def __init__(self, world, carla_map):
        self.world = world
        self.map = carla_map
        self.vehicle_bp = None

    def get_spawn_transform(self):
        transform = carla.Transform()
        transform.location.z = 0.25
        transform.location.x = 10.921
        transform.location.y = 3.5
        transform.rotation.yaw = 0
        wp = self.map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        transform = wp.transform
        transform.location.z = 0.10
        return transform

    def get_actor_bp(self):
        self.vehicle_bp = self.world.get_blueprint_library().find("vehicle.tesla.cybertruck")
        #self.vehicle_bp.set_attribute('color', '255,255,255')
        return self.vehicle_bp


def get_obps_position_x_y(position: str):
    positions = {"position_1": (-10.25, -11.00),
                 "position_2": (-9.80, 11.25),
                 "position_3": (10.60, 10.89),
                 "position_4": (9.40, -9.93)
                }
    if position in positions:
        return positions[position]
    else:
        raise KeyError("Cfg OBPS Invalid position")
