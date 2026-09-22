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

from typing import List, Dict, Union, Tuple
import carla
import shapely

from source.simulation_environment.agents.critical_region.critical_region import CriticalRegion
from source.simulation_environment.agents.critical_region.ego_prediction import EgoPrediction
from source.simulation_environment.agents.critical_region.target_prediction import TargetPrediction
from source.simulation_environment.agents.traj_shape import RoutePolygon
from source.simulation_environment.cfg.tactical_behavior_cfg import TacticalCfg
from source.simulation_environment.logs.tactical_log import RiskBlockLogItem, TacticalLog
from source.simulation_environment.obps.message import SensorMsg
import source.simulation_environment.logs.printl
import source.simulation_environment.utils as utils
from source.simulation_environment.agents.kalman_estimate import KalmanFilter
from source.simulation_environment.scenario.scenario_traj_manager import ScenarioTrajMng
from source.simulation_environment.agents.tactical_action import TacticalAction
from source.simulation_environment.agents.target_assumptions import TargetAssumption


class RiskBlock:
    def __init__(self,
                 target_id,
                 assumption: TargetAssumption,
                 ego_ref_speed: float,
                 ego_max_acc: float,
                 ego_max_break: float,
                 ego_len: float,
                 orig_ego_traj: RoutePolygon,
                 orig_target_traj: RoutePolygon,
                 sim_delta_t: float):
        target_critical_region = CriticalRegion(orig_target_traj)
        ego_critical_region = CriticalRegion(orig_ego_traj)
        target_critical_region.compute_critical_points(orig_ego_traj.polygon)
        ego_critical_region.compute_critical_points(orig_target_traj.polygon)
        #ego_critical_region.plot_regions(orig_target_traj.polygon)
        self.target_prediction: TargetPrediction = TargetPrediction(target_id, assumption, target_critical_region)
        self.ego_prediction: EgoPrediction = EgoPrediction(ego_ref_speed,ego_max_acc, ego_max_break, ego_len, ego_critical_region)

        self.target_id = target_id
        self.assumption: TargetAssumption = assumption
        self.kalman_f: KalmanFilter = KalmanFilter(0.001)
        self.msgs: List[SensorMsg] = list()
        self.msg = None
        self.aoi_ms = -1
        self.aoi_in_seconds = -1
        self.time_margin = 0.01
        # -- logging
        self.ego_ttlcr_less_adv_ttcr = -1
        self.ego_ttcr_greater_adv_ttlcr = -1
        self.ego_ttcr_less_adv_ttlcr = -1
        self.ego_go_pos = CriticalRegion.Position.UNKNOWN
        self.ego_pred_go_pos = CriticalRegion.Position.UNKNOWN
        # ---
        self.target_front_coord_x, self.target_front_coord_y = None, None

        self.reference_speed = ego_ref_speed
        self.risk_block_speed = ego_ref_speed
        self.ego_action = TacticalAction.CONTINUE
        self.target_acc = -1
        self.kalman_acc_value = -1


    def _get_target_acc(self, aoi_ms, velocity):

        self.kalman_acc_value = self.kalman_f.predict_acc(aoi_ms, velocity)
        if self.kalman_acc_value > self.assumption.max_acc:
            #print(f"kalman acc is higher: {self.kalman_acc_value}, assumption acc: {self.assumption.max_acc}")
            return self.kalman_acc_value

        return self.assumption.max_acc

    @staticmethod
    def _aoi_to_seconds(aoi) -> float:
        return aoi / 1_000

    def add_new_messages(self, sim_time, msgs: List[SensorMsg]):
        self.msgs.extend(msgs)
        # sort all the messages and
        self.msgs = sorted(self.msgs, key=lambda msg: msg.time_stamp, reverse=True)

    def get_latest_info(self):
        if len(self.msgs) == 0:
            return None

        return self.msgs[0]

    @staticmethod
    def stop_profile_speed_kmh(dist_to_cr_m: float,
                               v_now_mps: float,
                               v_max_mps:float,
                               a_des:float,
                               s_margin: float = 0.5,
                               ) -> float:
        """

        :param dist_to_cr_m: [m]     distance to critical region
        :param v_now_mps:    [m/s]   current velocity
        :param a_des:        [m/s²]  deceleration
        :param s_margin:     [m]     sensing/latency and the fact PID is not an ideal integrator
        :param v_max_mps:    [m/s]
        :return: velocity in km/h
        """
        # ideal “arrive with v=0 at CR” profile
        s = max(dist_to_cr_m - s_margin, 0.0)
        v_star = (2.0 * a_des * s) ** 0.5  # m/s
        v_star = min(v_star, v_max_mps)
        # optional: never command an acceleration above current speed on approach
        v_cmd_mps = min(v_star, v_now_mps)
        #USE the following code to limit speed variation
        # jerk_up = 1.0  # m/s per tick for increases
        # jerk_down = 3.0  # m/s per tick for decreases
        # v_cmd_mps = max(v_cmd_prev - jerk_down * dt, min(v_cmd_prev + jerk_up * dt, v_cmd_mps))
        return 3.6 * v_cmd_mps  # km/h

    def decision(self, sim_time, ego_vel: float, ego_front) -> Tuple[TacticalAction, float]:
        """
            # if we estimate that the target is inside or past the critical_region
            # we need to check if ego is before or after the critical_region, if it is inside we cannot do anything
            # 1) calculate the target time to leave the critical_region t_leave
            # 2) calculate the ego displacement in time t_leave for all ego strategies (GO, BREAK, ACCELERATE)
            # 3) check which strategy allows not to be in the critical_region
            # 4) if no strategy is found break
        :param sim_time:
        :param ego_vel:
        :param ego_front:
        :return: speed to follow in [km/h]
        """
        self.msg = self.get_latest_info()
        if self.msg is None:
            return TacticalAction.CONTINUE, self.reference_speed

        # --- RESET VARIABLES
        # reset time to cr variables
        self.ego_prediction.reset()
        self.target_prediction.reset()

        # Compute AoI
        self.aoi_ms = sim_time -  self.msg.time_stamp
        self.aoi_in_seconds = self._aoi_to_seconds(self.aoi_ms)

        # logging
        self.ego_pred_go_pos = CriticalRegion.Position.UNKNOWN
        self.ego_ttlcr_less_adv_ttcr = -1
        self.ego_ttcr_greater_adv_ttlcr = -1
        self.ego_ttcr_less_adv_ttlcr = -1

        # -- COMPUTE EGO POSITION, TIME TO CR, TIME TO LEAVE CR
        ego_front_p = shapely.Point(ego_front)
        shapely.prepare(ego_front_p)
        self.ego_prediction.predict(ego_vel, ego_front_p)

        # ---- COMPUTE TARGET POSITION, TIME TO CR, TIME TO LEAVE CR
        target_length =  self.msg.payload.measurement.length.value if self.msg.payload.measurement.length is not None else self.assumption.len
        target_front_p = shapely.Point( self.msg.payload.measurement.front.x,  self.msg.payload.measurement.front.y)
        shapely.prepare(target_front_p)
        self.target_acc = self._get_target_acc(self.aoi_ms, self.msg.payload.measurement.v.value)
        self.target_prediction.predict(self.aoi_in_seconds,  self.msg.payload.measurement.v.value, self.target_acc, target_length, target_front_p)

        # ---- COMPUTE CASES
        self.ego_action = TacticalAction.CONTINUE
        self.risk_block_speed = self.reference_speed
        # print("**********INITIAL, position: {0}".format(self.pose_to_string(self.target_pred_pos)) )

        # --- CASE target after the CR
        if self.target_prediction.current_pred_relative_pos == CriticalRegion.Position.AFTER_CR:
            # print("**********AFTER!!!")
            self.ego_action = TacticalAction.CONTINUE
            self.risk_block_speed = self.reference_speed

        # --- CASE target before the CR
        elif self.target_prediction.current_pred_relative_pos == CriticalRegion.Position.BEFORE_CR:

            # check where ego is
            if self.ego_prediction.current_relative_pos == CriticalRegion.Position.BEFORE_CR:
                # print("***********PPPPPPPPPPPPPPP!!!!!!")
                if self.ego_prediction.time_to_leave_cr <= self.target_prediction.time_to_cr - self.time_margin:
                    self.ego_action = TacticalAction.CONTINUE
                    self.risk_block_speed = self.reference_speed
                    self.ego_pred_go_pos = CriticalRegion.Position.AFTER_CR
                    self.ego_ttlcr_less_adv_ttcr = 1
                    # print(f"***********HERE 11  ego_time_to_leave_cr {self.ego_time_to_leave_cr}!!!!!!")
                elif self.ego_prediction.time_to_cr >= self.target_prediction.time_to_leave_cr + self.time_margin:
                    self.ego_action = TacticalAction.CONTINUE
                    self.risk_block_speed = self.reference_speed
                    self.ego_pred_go_pos = CriticalRegion.Position.BEFORE_CR
                    self.ego_ttcr_greater_adv_ttlcr = 1
                    # print(f"***********HERE 22 ego_time_to_cr { self.ego_time_to_cr}!!!!!!")
                else:
                    self.ego_action = TacticalAction.BREAKING
                    self.risk_block_speed = self.stop_profile_speed_kmh(
                        dist_to_cr_m=self.ego_prediction.dist_to_cr,
                        v_now_mps=ego_vel,
                        v_max_mps=self.reference_speed / 3.6,
                        a_des=self.ego_prediction.max_break,
                    )
                    self.ego_pred_go_pos = CriticalRegion.Position.INSIDE_CR
                    self.ego_ttlcr_less_adv_ttcr = 0
                    self.ego_ttcr_greater_adv_ttlcr = 0
                    # print("***********HERE 33!!!!!!")

            elif self.ego_prediction.current_relative_pos == CriticalRegion.Position.INSIDE_CR:
                # in case ego is inside the CR
                if self.ego_prediction.time_to_leave_cr <= self.target_prediction.time_to_cr - self.time_margin:
                    self.ego_action = TacticalAction.CONTINUE
                    self.risk_block_speed = self.reference_speed
                    # print("***********OOOOOOOOOOOOOOOO!!!!!!")
                else:
                    self.ego_action = TacticalAction.BREAKING
                    self.risk_block_speed = self.stop_profile_speed_kmh(
                        dist_to_cr_m=self.ego_prediction.dist_to_cr,
                        v_now_mps=ego_vel,
                        a_des=self.ego_prediction.max_break,
                        v_max_mps=self.reference_speed / 3.6
                    )
                    # print("***********SSSSSSSSSSSSSSSS!!!!!!")
            else:
                # ego is AFTER the CR
                self.ego_action = TacticalAction.CONTINUE
                self.risk_block_speed = self.reference_speed
                # print("***********TTTTTTTTTTTTTTT!!!!!!")

        # --- CASE target inside the CR
        elif self.target_prediction.current_pred_relative_pos == CriticalRegion.Position.INSIDE_CR:

            # check the time the target needs to move out from the CR
            if self.target_prediction.time_to_leave_cr == TargetPrediction.NO_TIME_TO_CR:
                # this should not be possible because target is INSIDE_CR
                print("[ERROR] (1) should not be possible!")
                self.ego_action = TacticalAction.CONTINUE
                self.risk_block_speed = self.reference_speed
                # print("***********AAAAAAA!!!!!!")
            else:
                # check the predicted ego position using the target time to leave the CR
                self.ego_pred_go_pos = self.ego_prediction.predict_go_relative_position(ego_vel,
                                                                                      self.target_prediction.time_to_leave_cr,
                                                                                      ego_front_p)

                if self.ego_prediction.current_relative_pos == CriticalRegion.Position.AFTER_CR:
                    self.ego_action = TacticalAction.CONTINUE
                    self.risk_block_speed = self.reference_speed
                    # print("***********BBBBBBBBBB!!!!!!")

                elif self.ego_prediction.current_relative_pos == CriticalRegion.Position.INSIDE_CR:
                    # could result in unavoidable crash (depending on dynamics and CR dimension)
                    self.ego_action = TacticalAction.BREAKING
                    self.risk_block_speed = self.stop_profile_speed_kmh(
                        dist_to_cr_m=self.ego_prediction.dist_to_cr,
                        v_now_mps=ego_vel,
                        a_des=self.ego_prediction.max_break,
                        v_max_mps=self.reference_speed / 3.6
                    )
                    # print("***********CCCCCCCC!!!!!!")

                elif self.ego_prediction.current_relative_pos == CriticalRegion.Position.BEFORE_CR:

                    if self.ego_prediction.time_to_cr <= self.target_prediction.time_to_leave_cr + self.time_margin:
                        # ego must break
                        self.ego_action = TacticalAction.BREAKING
                        self.risk_block_speed = 0.0
                        self.ego_ttcr_less_adv_ttlcr = 1
                    else:
                        # ego can continue
                        self.ego_action = TacticalAction.CONTINUE
                        self.risk_block_speed = self.reference_speed
                        self.ego_ttcr_less_adv_ttlcr = 0

                else:
                    # ego current pose is before the CR
                    print("[ERROR] we should not be here")

        # print("*************OUT")

        return self.ego_action, self.risk_block_speed

    def log(self, sim_time, world):
        ego_info = RiskBlockLogItem.EgoInfo(self.ego_prediction.cr.cn_orig_d,
                                            self.ego_prediction.cr.cf_orig_d,
                                            self.ego_prediction.d_front,
                                            self.ego_prediction.d_rear,
                                            self.ego_prediction.current_relative_pos,
                                            self.ego_go_pos,
                                            self.ego_prediction.dist_to_cr,
                                            self.ego_prediction.time_to_cr,
                                            self.ego_prediction.time_to_leave_cr,
                                            self.ego_ttlcr_less_adv_ttcr,
                                            self.ego_ttcr_greater_adv_ttlcr,
                                            self.ego_ttcr_less_adv_ttlcr,
                                            )

        actor = world.get_actor(self.target_id)
        points = utils.get_front_2d_box(actor)
        actor_front = shapely.Point(points)
        shapely.prepare(actor_front)
        front_gt, rear_gt = self.target_prediction._project_to_path(actor_front, actor.bounding_box.extent.x*2, 0)
        gt_rel_pos = self.target_prediction._get_relative_position(front_gt, rear_gt)

        target_info = RiskBlockLogItem.TargetInfo(
            self.target_prediction.cr.cn_orig_d,
            self.target_prediction.cr.cf_orig_d,
            self.target_prediction.d_front,
            self.target_prediction.d_rear,
            self.target_prediction.current_pred_relative_pos,
            self.target_prediction.dist_to_cr,
            self.target_prediction.time_to_cr,
            self.target_prediction.time_to_leave_cr,
            gt_rel_pos,
            front_gt,
            rear_gt
        )

        block_data = RiskBlockLogItem(sim_time,
                                      self.aoi_ms,
                                      self.risk_block_speed,
                                      self.ego_action,
                                      self.msg.sender,
                                      self.msg.id,
                                      ego_info,
                                      target_info
                                      )

        return block_data

class TacticalBehaviour:
    TAG = "[TacticalB]"

    def __init__(self,
                 ego: carla.Actor,
                 ego_max_acceleration: float,
                 ego_max_deceleration: float,
                 ego_target_speed: float,
                 tactical_cfg: TacticalCfg,
                 sim_delta_t: float,
                 scenario_trajectories: ScenarioTrajMng
                 ):

        self.ego = ego  # carla.Actor representing the vehicle
        self.world = ego.get_world()
        self.ego_reference_speed: float = ego_target_speed
        self.max_deceleration: float = ego_max_deceleration
        self.max_acceleration: float = ego_max_acceleration
        self.tactical_speed: float = ego_target_speed
        self.ego_len = ego.bounding_box.extent.x * 2
        self.printl = source.simulation_environment.logs.printl.PrintL(TacticalBehaviour.TAG, enabled=False)
        self.v_risk: float = 0
        self.risk_blocks: Union[Dict[RiskBlock], dict] = dict()
        self.sim_delta_t = sim_delta_t
        self.scenario_trajectories = scenario_trajectories
        self.tactical_cfg = tactical_cfg

    def risk_logic(self, sim_time) -> float:
        """
        :param sim_time:
        :return: speed to follow in km/h
        """
        risk_vector = [self.ego_reference_speed]

        fw = self.ego.get_transform().get_forward_vector()
        v = self.ego.get_velocity()
        #a = self.ego.get_acceleration()
        #a_fw = a.dot(fw)
        ego_vel = v.dot(fw)
        front = utils.get_front_2d_box(self.ego)

        for risk_block in self.risk_blocks.values():
            action, risk_speed = risk_block.decision(sim_time, ego_vel, front)
            risk_vector.append(risk_speed)

        self.tactical_speed = min(risk_vector)
        return self.tactical_speed

    def step(self, sim_time):
        """

        :param sim_time:
        :return: the velocity computed by the risk assessment block
        """

        return self.risk_logic(sim_time)

    @staticmethod
    def _data_association(messages: List[SensorMsg]) -> Dict:
        msg_map = dict()

        for message in messages:
            if message.payload.measurement.target_id not in msg_map:
                msg_map[message.payload.measurement.target_id] = list()
            msg_map[message.payload.measurement.target_id].append(message)

        return msg_map

    def set_ext_messages(self, messages: List[SensorMsg], sim_time):
        """
        Discard all messages that are -from the past-
        :param messages:
        :param sim_time: simulation time step
        :return:
        order the incoming message to get the freshest message!
        """

        msg_dict = self._data_association(messages)

        for key in msg_dict:
            if key not in self.risk_blocks:
                # if it is the fist time that we receive an observation for this agent id,
                # we need to create the risk block including the critical region
                # TODO
                # assumption should be created according to the cfg and the observed object,
                # use the observation semantic tag
                assumption = self.tactical_cfg.get_vehicle_assumption()

                risk_block = self.risk_blocks[key] = RiskBlock(key,
                                                               assumption,
                                                               self.ego_reference_speed,
                                                               self.max_acceleration,
                                                               self.max_deceleration,
                                                               self.ego_len,
                                                               self.scenario_trajectories.get_trajectory(self.ego.id),
                                                               self.scenario_trajectories.get_trajectory(key),
                                                               self.sim_delta_t)
            else:
                risk_block = self.risk_blocks[key]

            risk_block.add_new_messages(sim_time, msg_dict[key])

        self.printl.to_print("received messages: {0} sim_time: {1}".format(len(messages), sim_time))

    def log(self, sim_time):

        blocks_log: Dict[int, RiskBlockLogItem] = dict()

        for block in self.risk_blocks.values():
            block_data = block.log(sim_time, self.world)
            blocks_log[block.target_id] = block_data

        tactical_log = TacticalLog(tactical_ego_v=self.tactical_speed, blocks=blocks_log)

        return tactical_log
