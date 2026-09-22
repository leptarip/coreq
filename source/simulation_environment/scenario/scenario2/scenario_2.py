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
import math
import timeit

import numpy as np
import source.simulation_environment.vehicle_constants as constants

import source.simulation_environment.cfg.cfg_parser as cfg_parser
from source.simulation_environment.agents import traj_shape
from source.simulation_environment.agents.av_agent import AVAgent
from source.simulation_environment.agents.cav_agent import CAVAgent
from source.simulation_environment.agents.target_assumptions import TargetAssumption
from source.simulation_environment.cfg.tactical_behavior_cfg import TacticalCfg
from source.simulation_environment.cfg.vehicle_cfg import VehicleCfg
from source.simulation_environment.communication.communication import Communication
from source.simulation_environment.configuration_error import SimulationConfigurationError
from source.simulation_environment.scenario.base_scenario import BaseScenario
from source.simulation_environment.scenario.scenario_traj_manager import ScenarioTrajMng
from source.simulation_environment.obps.obps_manager import ObpsManager
from source.simulation_environment.logs.scenario_log import SimScenarioItem
from source.simulation_environment.scenario import scenario_roles
from source.simulation_environment.scenario.scenario2.scenario2_cfg_actors import EgoActor, AdversaryActor, StaticActor3, StaticActor1, StaticActor2, get_obps_position_x_y
import source.simulation_environment.logs.printl
import carla
import source.simulation_environment.utils as utils
import source.simulation_environment.logs.redis_data as rd
from source.simulation_environment import vehicle_constants

SCENARIO_2_NAME = "intersection2"

class ScenarioIntersection2(BaseScenario):
    """
    A 4 way intersection.
    CAV (ego) goes straight from Nord to South
    ADV goes straight from Est to West
    crossing ego's trajectory and burning the red.
    """
    TAG = "Scenario:Inter2"

    def __init__(self,
                 experiment_name: str,
                 world: carla.World,
                 carla_map: carla.Map,
                 cfg: dict,
                 delta_time,
                 sampler=None,
                 seed=None):
        super().__init__(experiment_name,world, carla_map, cfg, delta_time)
        self.ego = None
        self.adv = None
        self.ego_actor = EgoActor(self.world, self.carla_map, self.cfg["ego"])
        self.adv_actor = AdversaryActor(self.world, self.carla_map, self.cfg["adversary"])
        self.scenario_traj_manager = ScenarioTrajMng()
        self.ego_agent = None
        self.adv_agent = None
        self.static1 = None
        self.static2 = None
        self.static3 = None
        self.static_actor1 = StaticActor1(self.world, self.carla_map)
        self.static_actor2 = StaticActor2(self.world, self.carla_map)
        self.static_actor3 = StaticActor3(self.world, self.carla_map)
        self.obps_manager = None
        self.communication = None
        self.sampler = sampler
        self.seed = seed
        self.list_buildings = None
        self.printl = source.simulation_environment.logs.printl.PrintL(self.TAG, enabled=True)
        self.log_time = []
        self.obs_time = []


    def _create_ego_agent(self, cfg: dict, experiment_name: str, vehicle: carla.Actor, map_buildings, scenario_traj):
        try:
            experiment = rd.SimRunKeys(experiment_name, scenario_roles.ROLE_HERO)

            v_cfg = VehicleCfg(
                       max_control_break=self.ego_actor.get_control_max_break(),
                       max_control_throttle=self.ego_actor.get_control_max_throttle(),
                       max_acceleration=self.ego_actor.get_max_acceleration(),
                       max_deceleration=self.ego_actor.get_max_deceleration(),
                       target_speed=cfg["target_v"],
                       sensor_range=cfg["perception"]["fov"]["range"],
                       sensor_resolution=cfg["perception"]["fov"]["ray_step"],
                       sensor_fov=cfg["perception"]["fov"]["degrees"],
                       prediction_window=cfg["perception"]["prediction_window"],
                       initial_speed=0,
                       role_name=cfg["role_name"],
                       decision_period=cfg["decision_period"],
                       stream=False
                       )

            # In this scenario the adv is a vehicle
            assumption = TargetAssumption(max_speed=50, #this scenario max speed for the adversary
                                   max_break=constants.MAX_BRAKE,
                                   max_acc=constants.DEFAULT_TARGET_A,
                                   agent_type=TargetAssumption.TargeType.VEHICLE,
                                   target_len=vehicle_constants.DEFAULT_VEHICLE_LENGTH)

            tactical = TacticalCfg()
            tactical.add_assumption(assumption)

            self.ego_agent = CAVAgent(
                vehicle=vehicle,
                cfg=v_cfg,
                tactical_cfg=tactical,
                experiment=experiment,
                map_buildings=map_buildings,
                sim_delta_t=self.delta_time,
                scenario_trajectories=scenario_traj
            )

        except Exception as e:
            self.printl.to_print(header="ERROR",
                                 message="parsing ego config: {0}".format(e),
                                 force=True,
                                 where=self.printl.STD_OUT_AND_MEM)
            return False

        return True

    def _create_adv_agent(self, cfg: dict, experiment_name: str, vehicle: carla.Actor, carla_map, map_buildings):
        try:
            experiment = rd.SimRunKeys(experiment_name, scenario_roles.ROLE_ADVERSARY)

            v_cfg = VehicleCfg(
                       max_control_break=self.adv_actor.get_control_max_break(),
                       max_control_throttle=self.adv_actor.get_control_max_throttle(),
                       max_acceleration=self.adv_actor.get_max_acceleration(),
                       max_deceleration=self.adv_actor.get_max_deceleration(),
                       target_speed=cfg["target_v"],
                       sensor_range=cfg["perception"]["fov"]["range"],
                       sensor_resolution=cfg["perception"]["fov"]["ray_step"],
                       sensor_fov=cfg["perception"]["fov"]["degrees"],
                       prediction_window=cfg["perception"]["prediction_window"],
                       initial_speed=0,
                       role_name=cfg["role_name"],
                       decision_period=cfg["decision_period"],
                       stream=False
                       )

            self.adv_agent = AVAgent(
                vehicle= vehicle,
                cfg= v_cfg,
                experiment=experiment,
                map_buildings=map_buildings,
                map_inst = carla_map)

        except Exception as e:
            self.printl.to_print(header="ERROR",
                                 message="error parsing adversary config: {0}".format(e),
                                 force=True,
                                 where=self.printl.STD_OUT_AND_MEM)
            return False

        return True

    def _create_all_obps(self, cfg):

        try:
            sensors = cfg_parser.get_sensors_cfg(cfg, get_obps_position_x_y)
            re_cfg = cfg.get("rare_event", {})
            rare_event_enabled = re_cfg.get("enabled", False)
            sensor_bias = re_cfg.get("sensor_bias", None) if rare_event_enabled else None
            sampler = self.sampler if rare_event_enabled else None

            if len(sensors) < 1:
                return False

            self.obps_manager = ObpsManager(
                self.experiment_name,
                sensors,
                self.communication,
                self.adv,
                self.delta_time,
                sensor_bias=sensor_bias,
                sampler=sampler,
                seed=self.seed)
        except SimulationConfigurationError:
            raise
        except Exception as e:
            self.printl.to_print(message="error creating sensors: {0}".format(e))
            return False
        return True

    def _create_communication(self, cfg):

        re_cfg = cfg.get("rare_event", {})
        bias_cfg = re_cfg if re_cfg.get("enabled", False) else None
        self.communication = Communication(self.delta_time,
                                           self.experiment_name,
                                           cfg["communication"],
                                           sampler=self.sampler if re_cfg.get("enabled", False) else None,
                                           bias_cfg=bias_cfg)
        return True

    def generate_scenario(self) -> BaseScenario.ErrorCode:

        # map building list, pinned to a deterministic order: the server does not
        # guarantee a stable ordering between processes, and it feeds STRtree indices.
        self.list_buildings = utils.sort_environment_objects(
            self.world.get_environment_objects(carla.CityObjectLabel.Buildings))

        # spawn ego vehicle
        self.ego = self.world.spawn_actor(self.ego_actor.get_actor_bp(),
                                self.ego_actor.get_spawn_transform())
        # spawn adversary vehicle
        self.adv = self.world.spawn_actor(self.adv_actor.get_actor_bp(),
                                self.adv_actor.get_spawn_transform())

        self.static1 = self.world.spawn_actor(self.static_actor1.get_actor_bp(),
                                      self.static_actor1.get_spawn_transform())
        self.static2 = self.world.spawn_actor(self.static_actor2.get_actor_bp(),
                                         self.static_actor2.get_spawn_transform())
        self.static3 = self.world.spawn_actor(self.static_actor3.get_actor_bp(),
                                              self.static_actor3.get_spawn_transform())

        err_3 = self._create_communication(self.cfg)
        err_4 = self._create_all_obps(self.cfg)

        # create ego and adv agents
        err_1 = err_2 = True

        if err_1 is False or err_2 is False or err_3 is False or err_4 is False:
            return BaseScenario.ErrorCode.ERROR_LOADING_CFG

        # crate sensors and pass the target
        #self.obps_manager.create_sensors(target=self.adv)
        self.obps_manager.set_building_list(self.list_buildings)

        return BaseScenario.ErrorCode.OK

    def pre_run(self) -> dict:
        v = 1000
        while v > 3:
            v = self.ego.get_velocity()
            v = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
            self.world.tick(100)
        list_vehicle = self.world.get_actors().filter("*vehicle*")
        list_obs_ego = [v for v in list_vehicle if v.id != self.ego.id]
        list_obs_adversary = [v for v in list_vehicle if v.id != self.adv.id]

        err_1 = self._create_adv_agent(cfg=self.cfg["adversary"],
                                       experiment_name=self.experiment_name,
                                       vehicle=self.adv,
                                       carla_map=self.carla_map,
                                       map_buildings=self.list_buildings)

        err_2 = self._create_ego_agent(self.cfg["ego"],
                                       experiment_name=self.experiment_name,
                                       vehicle=self.ego,
                                       map_buildings=self.list_buildings,
                                       scenario_traj= self.scenario_traj_manager)

        # set routes
        self.adv_agent.set_destination(self.adv_actor.get_destination())
        self.ego_agent.set_destination(self.ego_actor.get_destination())

        # The world is in synchronous mode, so world.tick() already blocks until the
        # server has finished the frame. These ticks settle the scene and must stay;
        # the wall-clock sleep that used to sit beside them synchronised nothing.
        for i in range(50):
            self.world.tick(100)

        adv_path_polygon = self.adv_agent.get_route_polygon()
        self.scenario_traj_manager.add_item(self.adv_agent.get_actor().id, adv_path_polygon)
        ego_path_polygon = self.ego_agent.get_route_polygon()
        self.scenario_traj_manager.add_item(self.ego_agent.get_actor().id, ego_path_polygon)

        # set initial speeds
        #v = self.ego.get_transform().get_forward_vector() * self.ego_actor.get_initial_speed()
        #self.ego.set_target_velocity(v)

        #v = self.adv.get_transform().get_forward_vector() * self.adv_actor.get_initial_speed()
        #self.adv.set_target_velocity(v)

        self.draw_sensor_positions()

        extra = {"list_actors": list_vehicle, "list_obs_ego": list_obs_ego, "list_obs_adversary": list_obs_adversary}

        return extra

    def is_scenario_done(self, sim_time) -> BaseScenario.State:
        ego_done = self.ego_agent.done()
        #adv_done = self.adv_agent.done()
        if ego_done is True:
            key = rd.get_h_key(self.experiment_name, rd.K_SCENARIO_TERMINATION)
            stream_data_name = rd.get_stream_name(self.experiment_name, key.decode())
            log_data = SimScenarioItem(SCENARIO_2_NAME, sim_time, "agent_done", self.delta_time)
            rd.save_stream_scenario_termination(stream_data_name, log_data)
            return BaseScenario.State.END_DONE

        ego_route_poly = self.ego_agent.get_route_polygon()
        adv_route_poly = self.adv_agent.get_route_polygon()

        if ego_route_poly is not None and adv_route_poly is not None:
            rs = utils.are_shapes_colliding(ego_route_poly.actor_polygon, adv_route_poly.actor_polygon)
            if rs is True:
                key = rd.get_h_key(self.experiment_name, rd.K_SCENARIO_TERMINATION)
                stream_data_name = rd.get_stream_name(self.experiment_name, key.decode())
                log_data = SimScenarioItem(SCENARIO_2_NAME, sim_time, "col_shapes", self.delta_time)
                rd.save_stream_scenario_termination(stream_data_name, log_data)
                return BaseScenario.State.END_COLLISION

        return BaseScenario.State.IN_PROGRESS

    def run_step(self, sim_time, extra) -> BaseScenario.State:
        # termination condition
        scenario_state = self.is_scenario_done(sim_time)
        if scenario_state != BaseScenario.State.IN_PROGRESS:
            np_log_time = np.asarray(self.log_time)
            np_obs_time = np.asarray(self.obs_time)
            self.printl.to_print(message="##### log[avg: {0}s, min {1}s, max{2}s ] obs [avg: {3}s min {4}s, max{5}s @ "
                                         "index={6} ]#######"
                                 .format(np.average(np_log_time),
                                         np.min(np_log_time),
                                         np.max(np_log_time),
                                         np.average(np_obs_time),
                                         np.min(np_obs_time),
                                         np.max(np_obs_time),
                                         np.argmax(np_obs_time)
                                         ))
            return scenario_state

        # sensors observation
        self.obps_manager.tick(sim_time, extra["list_actors"])
        self.communication_tick(sim_time)

        # ego collects external information
        messages = self.communication.get_messages(receiver=self.ego_agent.name)

        # set messages
        self.ego_agent.set_ext_messages(messages, sim_time)

        start_1 = timeit.default_timer()

        # pass adv as target to ego_Agent
        target = traj_shape.Target(self.adv_agent.get_actor(), self.adv_agent.get_route_polygon())
        self.ego_agent.set_target_to_observe(target)

        # pass ego as target to adv_agent
        e = traj_shape.Target(self.ego_agent.get_actor(), self.ego_agent.get_route_polygon())
        self.adv_agent.set_target_to_observe(e)

        self.ego_agent.step(sim_time, extra["list_obs_ego"])
        self.adv_agent.step(sim_time, extra["list_obs_adversary"])
        self.obs_time.append(timeit.default_timer() - start_1)

        start_1 = timeit.default_timer()
        # save ego stats
        self.ego_agent.log(sim_time)
        self.adv_agent.log(sim_time)
        self.log_time.append(timeit.default_timer() - start_1)

        return BaseScenario.State.IN_PROGRESS

    def sensors_tick(self, sim_time):
        for s in self.obps_manager.sensor_list:
            s.sensor_tick(sim_time)

    def communication_tick(self, sim_time):
        self.communication.comm_tick(sim_time)

    def get_ego_agent(self):
        return self.ego_agent

    def get_adv_agent(self):
        return self.adv_agent

    @staticmethod
    def get_scenario_map() -> str:
        return "sim01"

    def destroy(self):
        self.destroy_actors(
            self.ego,
            self.adv,
            self.static1,
            self.static2,
            self.static3,
        )


    def draw_sensor_positions(self):
        for sensor in self.obps_manager.sensor_list:
            position = sensor.get_position()
            self.world.debug.draw_box(
                carla.BoundingBox(carla.Location(x=position[0], y=position[1]), carla.Vector3D(0.5, 0.5, 4)),
                carla.Rotation(yaw=sensor.get_orientation()),
                0.1,
                carla.Color(0, 0, 0, 0),
                0)


    @staticmethod
    def get_stop_lane_points():
        return [(11, 0), (11, -5.5)]

    def draw_stop_line(self):
        lt = 0

        lane = self.get_stop_lane_points()

        self.world.debug.draw_line(
            carla.Location(x=lane[0][0], y=lane[0][1]),
            carla.Location(x=lane[1][0], y=lane[1][1]),
            thickness=0.5,
            color=carla.Color(0, 0, 0), life_time=lt)
