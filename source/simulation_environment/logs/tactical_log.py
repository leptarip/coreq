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
from json import JSONEncoder
from typing import Dict


class RiskBlockLogItem:
    class EgoInfo:
        def __init__(self,
                     ego_cr_cn,
                     ego_cr_cf,
                     gt_front,
                     gt_rear,
                     gt_current_pos,
                     pred_go_position,
                     dist_to_cr,
                     ttcr,
                     ttlcr,
                     ego_ttlcr_less_adv_ttcr,
                     ego_ttcr_greater_adv_ttlcr,
                     ego_ttcr_less_adv_ttlcr
                     ):
            """

            :param ego_cr_cn:  critical region - critical point near
            :param ego_cr_cf:  critical region - critical point far
            :param gt_front:   ground truth ego front
            :param gt_rear:    ground truth ego rear
            :param gt_current_pos:    ground truth ego position
            :param pred_go_position: predicted ego position with target ttcr or ttlcr
            :param dist_to_cr: distance to critical region
            :param ttcr:  tie to critical region
            :param ttlcr: time to leave critical region
            :param ego_ttlcr_less_adv_ttcr:
            :param ego_ttcr_greater_adv_ttlcr:
            :param ego_ttcr_less_adv_ttlcr:
            """
            self.cr_cn = ego_cr_cn
            self.cr_cf = ego_cr_cf
            self.gt_front = gt_front
            self.gt_rear = gt_rear
            self.gt_current_pos = gt_current_pos
            self.pred_go_pos = pred_go_position
            self.dist_to_cr = dist_to_cr
            self.ttcr = ttcr
            self.ttlcr = ttlcr
            self.ttlcr_less_adv_ttcr = ego_ttlcr_less_adv_ttcr
            self.ttcr_greater_adv_ttlcr = ego_ttcr_greater_adv_ttlcr
            self.ttcr_less_adv_ttlcr = ego_ttcr_less_adv_ttlcr

    class TargetInfo:
        def __init__(self,
                     target_cr_cn,
                     target_cr_cf,
                     pred_front,
                     pred_rear,
                     pred_rel_pos,
                     dist_to_cr,
                     ttcr,
                     ttlcr,
                     gt_rel_pos,
                     gt_front,
                     gt_rear):
            """

            :param target_cr_cn:  critical region - critical point near
            :param target_cr_cf:  critical region - critical point far
            :param pred_front:    predicted target front
            :param pred_rear:     predicted target rear
            :param pred_rel_pos:  predicted target relative position
            :param dist_to_cr:    distance to critical region
            :param ttcr:          time to critical region
            :param ttlcr:         time to leave critical region
            :param gt_rel_pos:    ground truth target relative position
            :param gt_front:      ground truth target front
            :param gt_rear:       ground truth target rear
            """
            self.cr_cn = target_cr_cn
            self.cr_cf = target_cr_cf
            self.dist_to_cr = dist_to_cr
            self.ttcr = ttcr
            self.ttlcr = ttlcr
            self.pred_front = pred_front
            self.pred_rear = pred_rear
            self.pred_rel_pos = pred_rel_pos
            self.gt_rel_pos = gt_rel_pos
            self.gt_front = gt_front
            self.gt_rear = gt_rear

    def __init__(self,
                 sim_time,
                 aoi,
                 ego_v_decision,
                 ego_action,
                 msg_sender,
                 msg_id,
                 ego_info: EgoInfo,
                 target_info: TargetInfo):
        self.sim_time = sim_time
        self.aoi = aoi
        self.v_block = ego_v_decision
        self.ego_action = ego_action
        self.msg_sender = msg_sender
        self.msg_id = msg_id
        self.ego_info = ego_info
        self.target_info = target_info


class RiskBlockLogItemEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class TacticalLog:
    def __init__(self, tactical_ego_v: float, blocks: Dict[int, RiskBlockLogItem]):
        self.v_tactical = tactical_ego_v
        self.blocks = blocks


class TacticalLogEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__
