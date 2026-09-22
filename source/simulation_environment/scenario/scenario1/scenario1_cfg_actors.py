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
        self.init_position_x = -47
        self.ini_position_y = 19
        self.destination_x = -70
        self.destination_y = -1

    def get_spawn_transform(self):
        transform = carla.Transform()
        transform.location.z = 0.10
        transform.location.x = self.init_position_x
        transform.location.y = self.ini_position_y
        transform.rotation.yaw = 0
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
        return 0.48
        #return 0.9

    @staticmethod
    def get_control_max_break() -> float:
        """
        To be used with carla PID controller to limit the max deceleration
        :return:
        """
        return 0.40
        # 0.48
        #return 0.5

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
        need to find it out with experimentation and by tweaking your implemented controller
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
        self.position_x = -54
        self.position_y = -22
        self.destination_x = -54
        self.destination_y = 15 #original destination_y is 15
        self.initial_v = 0#cfg["initial_v"]
        self.role_name = cfg["role_name"]
        self.target_v = cfg["target_v"]

    def get_spawn_transform(self):
        transform = carla.Transform()
        transform.location.z = 0.10
        transform.location.x = self.position_x
        transform.location.y = self.position_y
        transform.rotation.yaw = 90
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
        need to find it out with experimentation and by tweaking your implemented controller
        :return:
        """
        return 4.0

    @staticmethod
    def get_max_deceleration() -> float:
        """
        need to find it out with experimentation and by tweaking your implemented controller
        :return:
        """
        # the real one is 2.16!
        # use another value to create errors in calculations
        return 2.16 #2.75

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


class DeliveryVanActor1:
    """
    This is the van to the left of ego
    """

    def __init__(self, world, carla_map):
        self.world = world
        self.map = carla_map
        self.vehicle_bp = None

    def get_spawn_transform(self):
        transform = carla.Transform()
        transform.location.z = 0.25
        transform.location.x = -51
        transform.location.y = -13.5
        wp = self.map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        transform = wp.transform
        transform.location.z = 0.10
        return transform

    def get_actor_bp(self):
        self.vehicle_bp = self.world.get_blueprint_library().find('vehicle.carlamotors.carlacola')
        return self.vehicle_bp

    def get_target_destination(self):
        location = carla.Location()
        location.x = -51
        location.y = -4
        wp = self.map.get_waypoint(location, project_to_road=False, lane_type=carla.LaneType.Driving)
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

        wp = self.map.get_waypoint(self.get_target_destination(),
                                   project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        wp.transform.rotation.yaw = yaw
        plan.append((wp, RoadOption.STRAIGHT))

        return plan


class DeliveryVanActor2:
    """
    This is the van in front of ego
    """

    def __init__(self, world, carla_map):
        self.world = world
        self.map = carla_map
        self.vehicle_bp = None

    def get_spawn_transform(self):
        transform = carla.Transform()
        transform.location.z = 0.25
        transform.location.x = -54
        transform.location.y = -9.0 # original position-10.49
        wp = self.map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        transform = wp.transform
        transform.location.z = 0.10
        return transform

    def get_actor_bp(self):
        self.vehicle_bp = self.world.get_blueprint_library().find('vehicle.carlamotors.carlacola')
        return self.vehicle_bp

    def get_target_destination(self):
        location = carla.Location()
        location.x = -54
        location.y = 46
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

        plan = [(wp, RoadOption.STRAIGHT)]

        wp = self.map.get_waypoint(self.get_target_destination(),
                                   project_to_road=True,
                                   lane_type=carla.LaneType.Driving)

        plan.append((wp, RoadOption.STRAIGHT))

        return plan

def get_position_x_y(position: str):
    positions = {"position_1": (-47.10, -9.50),
                 "position_2": (-38.60, 1.00),
                 "position_3": (-59.10, 19.00)
                }
    if position in positions:
        return positions[position]
    else:
        raise KeyError("Cfg OBPS Invalid position")
