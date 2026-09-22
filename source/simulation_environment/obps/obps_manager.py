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

from typing import Dict, List, Optional

import carla
import shapely

from source.simulation_environment.communication.communication import Communication
from source.simulation_environment.obps.obps import Sensor
from source.simulation_environment.cfg.sensor_cfg import SensorCfg
import source.simulation_environment.utils as utils
from source.simulation_environment.logs import redis_data as rd
from source.simulation_environment.rare_events.wrapper import ImportanceSampler


class ObpsManager:
    def __init__(self,
                 exp_name,
                 cfg_list: List[SensorCfg],
                 comm_medium: Communication,
                 target: carla.Actor,
                 delta_time,
                 sensor_bias: Optional[dict] = None,
                 sampler: Optional["ImportanceSampler"] = None,
                 seed: Optional[int] = None
                 ):
        self.cfg_list = cfg_list
        self.sensor_list = list()
        self.building_list = list()
        self.building_shapes = list()
        self.actor_shapes: List[utils.UtilPair] = list()
        # Built once per tick and shared by every sensor. Each sensor used to
        # rebuild an ActorShapeMap over all actors just to look up its one target.
        self.shape_by_actor_id: Dict[int, "shapely.Polygon"] = dict()
        self.comm_medium = comm_medium
        self.target = target
        self.ext_msg_sensor_key = rd.get_stream_name(exp_name, rd.get_h_key(exp_name, rd.K_EXT_SENSOR_MSG).decode())
        self.sensor_bias = sensor_bias
        self.sampler = sampler
        if target is not None:
            for cfg in self.cfg_list:
                self.sensor_list.append(Sensor(self.ext_msg_sensor_key,
                                               cfg,
                                               comm_medium,
                                               target,
                                               delta_time,
                                               sensor_bias=sensor_bias,
                                               sampler=self.sampler,
                                               seed=seed))

    def set_building_list(self, building_list):
        self.building_list = building_list
        for b in self.building_list:
            cords = utils.create_bb_points_2D(b.bounding_box)
            points = (
                (cords[0, 0], cords[0, 1]),
                (cords[1, 0], cords[1, 1]),
                (cords[2, 0], cords[2, 1]),
                (cords[3, 0], cords[3, 1])
            )
            polygon = shapely.Polygon(points)
            shapely.prepare(polygon)
            self.building_shapes.append(polygon)

    def create_actor_shapes(self, actors):
        for actor in actors:
            world_coords = utils.create_bb_shape(actor)
            polygon = shapely.Polygon((
                (world_coords[0, 0], world_coords[1, 0]),
                (world_coords[0, 1], world_coords[1, 1]),
                (world_coords[0, 2], world_coords[1, 2]),
                (world_coords[0, 3], world_coords[1, 3])
            ))
            shapely.prepare(polygon)
            self.actor_shapes.append(utils.UtilPair(actor, polygon))
            self.shape_by_actor_id[actor.id] = polygon

    def tick(self, sim_time, actor_list):
        self.actor_shapes.clear()
        self.shape_by_actor_id.clear()
        self.create_actor_shapes(actor_list)

        for sensor in self.sensor_list:
            sensor.sensor_tick(sim_time, self.actor_shapes, self.shape_by_actor_id)
