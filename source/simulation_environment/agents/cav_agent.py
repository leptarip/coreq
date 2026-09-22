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
from typing import List

import carla

import source.simulation_environment.agents.traj_shape as tjs
from source.simulation_environment.agents.av_agent import AVAgent
from source.simulation_environment.agents.tactical_behaviour import TacticalBehaviour
from source.simulation_environment.cfg.tactical_behavior_cfg import TacticalCfg
from source.simulation_environment.cfg.vehicle_cfg import VehicleCfg
import source.simulation_environment.logs.redis_data as rd
from source.simulation_environment.logs.ego_logs import MsgLogItem, PerceptionLogItem
from source.simulation_environment.obps.message import SensorMsg


class CAVAgent(AVAgent):
    """
    CAVAgent implements an agent that navigates the scene and as 2D perception and longitudinal object avoidance.
    plus receives messages from infrastructure and makes decision
    """

    def __init__(self,
                 vehicle: carla.Actor,
                 cfg: VehicleCfg,
                 tactical_cfg: TacticalCfg,
                 experiment,
                 map_buildings,
                 sim_delta_t,
                 scenario_trajectories,
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
        super().__init__(vehicle, cfg, experiment, map_buildings, opt_dict, map_inst, grp_inst)
        self.tactical_behaviour = TacticalBehaviour(vehicle,
                                                    cfg.max_acceleration,
                                                    cfg.max_deceleration,
                                                    cfg.target_speed,
                                                    tactical_cfg,
                                                    sim_delta_t,
                                                    scenario_trajectories
                                                    )
        self.v_tact = 0
        self.messages : List[SensorMsg] = list()
        self.dist_to_target = -1

    def step(self, sim_time, list_obs=None) -> None:
        """Execute one step of navigation and move ego."""
        if sim_time % self.ad_period == 0:
            self.route_polygon = tjs.get_route_polygon(self._vehicle, self._local_planner.get_plan(),
                                                       cache=self._route_cache)

            # update the perception sensor according to new vehicle position
            self._update_sensor()
            # pass all relevant ground truth carla actors to perception
            self.perception.set_obstacles(list_obs)
            self.perception.step()
            # detect all obstacles intersecting the vehicle's path
            self.dist_to_obj = self.detection.step(self.perception.fov_polygon,
                                                   self.route_polygon,
                                                   self.perception.actors_map,
                                                   self.target)
            # get distance to target if in fov
            self.dist_to_target = self.detection.get_dist_to_target_in_fov(self.perception.fov_polygon,
                                                                           self.route_polygon,
                                                                           self.target)

            self.v_ctl, _, _ = self.lv_ctl.compute_velocity(self.dist_to_obj, margin_distance=1.0)

            # get risk velocity from tactical behavior
            self.v_tact = self.tactical_behaviour.step(sim_time) / 3.6

            # check if there are detected obstacles within fov
            # 1) if yes then apply the constraint on velocity
            # 2) else apply the target speed or tactical velocity
            if self.detection.obs_detected is True:
                self.v_selected = min(self.reference_speed, self.v_ctl, self.v_tact)
            else:
                self.v_selected = min(self.reference_speed, self.v_tact)

            # set the new reference speed to be followed by low level controlled (PID)
            self.set_target_speed(self.v_selected * 3.6)
            control = self._local_planner.run_step()
            self._vehicle.apply_control(control)

    def set_ext_messages(self, messages: List[SensorMsg], sim_time):
        self.messages = messages
        self.tactical_behaviour.set_ext_messages(messages, sim_time)

    def log(self, sim_time):
        """
        should be called after the agent step is called
        :param sim_time:
        :return:
        """
        super().log(sim_time)

        #log tactical information
        if sim_time % self.ad_period == 0:
            tactical_log = self.tactical_behaviour.log(sim_time)
            perception_log = PerceptionLogItem(sim_time, self.target.actor.id, self.dist_to_target)
            rd.save_stream_ego_tactical(self.experiment.stream_tactical_name, tactical_log)
            rd.save_stream_ego_perception(self.experiment.stream_perception_name, perception_log)

        #log all new received messages
        for msg in self.messages:
            rd.save_stream_ego_msg(self.experiment.stream_obps_msg_name, MsgLogItem(sim_time, msg.sender, msg.id))

