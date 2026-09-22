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
import random

import carla
import shapely
import math

import shapely.affinity
from source.simulation_environment.cfg.sensor_cfg import SensorCfg
import source.simulation_environment.utils as utils
from source.simulation_environment.obps.logic.sector_logic import FovSector, SectorLogic
from source.simulation_environment.obps.measurement import Measurement, MeasuredPoint, MeasuredValue, Measured2dBox
from source.simulation_environment.obps.sensor_prob import SensorProb
from source.simulation_environment.obps.sensor_result import SensorResult


class FovSectorStaticGhost(FovSector):
    SCALE = 0.75
    RADIUS = 50
    PROB_DET = 1
    PROB_MISS = 0
    PROB_GHOST = 0
    WRONG = 0

    def __init__(self, cfg: dict):
        super().__init__(cfg, None, fov_type=SensorCfg.TYPE_SECTOR_GHOST)


class SectorStaticGhost:
    TAG = "[SectorStaticGhost] "
    MAX_SENSOR_FOV = 360

    def __init__(self, cfg: FovSectorStaticGhost, seed=None):

        self.target_shape = None
        self.target_in_A = False
        self.target_in_B = False
        self.cfg = cfg
        self.sensor_probA = SensorProb(cfg.probA_det, cfg.probA_miss, cfg.probA_ghost, cfg.probA_wrong)
        self.sensor_probB = SensorProb(cfg.probB_det, cfg.probB_miss, cfg.probB_ghost, cfg.probB_wrong)
        self.actor_sh_map = utils.ActorShapeMap()
        points = self.compute_fov_sector()
        # polygon_B is the external polygon
        self.fov_polygon_B = self.create_fov_shape(points)
        # polygon_A is the internal polygon
        angle = math.radians(self.cfg.orientation)
        shift = (0.8 * math.cos(angle), 0.8 * math.sin(angle), 0)
        self.fov_polygon_A = shapely.affinity.scale(self.fov_polygon_B, xfact=0.75, yfact=0.75,
                                                    origin=self.cfg.position)
        self.fov_polygon_A = shapely.affinity.translate(self.fov_polygon_A, xoff=shift[0], yoff=shift[1])
        shapely.prepare(self.fov_polygon_A)
        # get difference B - A polygon
        self.fov_difference = self.fov_polygon_B.difference(self.fov_polygon_A)
        shapely.prepare(self.fov_difference)

        self.weights_A = [self.cfg.probA_det, self.cfg.probA_wrong, self.cfg.probA_miss]
        self.weights_B = [self.cfg.probB_det, self.cfg.probB_wrong, self.cfg.probB_miss]
        self.choices = [SensorResult.DETECTION, SensorResult.WRONG, SensorResult.MISS]
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)

    def compute_fov_sector(self):
        """

        :param self:
        :return: list of shapely points defining the fov polygon
        """
        origin = shapely.Point(self.cfg.position[0], self.cfg.position[1])

        start = end = self.cfg.orientation
        if self.cfg.fov_degrees < 360:
            angle = self.cfg.fov_degrees // 2
            start = start - angle
            end = end + angle

        steps = self.cfg.fov_degrees

        ideal_poly_points = []

        for step in range(steps + 1):
            angle = math.radians(start + step)
            x = self.cfg.radius * math.cos(angle) + origin.x
            y = self.cfg.radius * math.sin(angle) + origin.y
            # creating the ray=(center, FOV exterior point)
            point_step = shapely.Point(x, y)
            ideal_poly_points.append(point_step)

        if self.cfg.fov_degrees < SectorLogic.MAX_SENSOR_FOV:
            # add the central point to connect the FOV polygon to the center of the obps
            ideal_poly_points.append(origin)
        else:
            ideal_poly_points.append(ideal_poly_points[0])

        return ideal_poly_points

    @staticmethod
    def create_fov_shape(points):
        """
       :param points: list of shapely points forming the fov polygon
       :return: the fov_polygon
       """
        fov_polygon = shapely.Polygon(tuple(points))
        shapely.prepare(fov_polygon)

        return fov_polygon

    def _get_ghost_measurement(self, fov, target) -> Measurement:
        vel = 12  # m/s
        unc = 0
        if self.cfg.unc_vel != 0:
            unc = random.normalvariate(0, self.cfg.unc_vel)
            vel = vel + unc

        point = shapely.Point([5 + unc, -3.5 + unc])
        fw = target.get_transform().get_forward_vector()

        front_x = point.x + unc + target.bounding_box.extent.x * fw.x
        front_y = point.y + unc + target.bounding_box.extent.x * fw.y
        p1, p2, p3, p4 = utils.get_2d_box_points(point,
                                                 target.bounding_box.extent,
                                                 fw,
                                                 target.get_transform().get_right_vector())

        return Measurement(
            target_id=-1,
            position=MeasuredPoint(point.x, point.y, uncertainty=self.cfg.unc_pos),
            front=MeasuredPoint(front_x, front_y, uncertainty=self.cfg.unc_pos),
            v=MeasuredValue(vel, uncertainty=self.cfg.unc_vel),
            length=target.bounding_box.extent.x,
            width=target.bounding_box.extent.y,
            box2d=Measured2dBox(p1, p2, p3, p4, uncertainty=self.cfg.unc_pos)
        )

    def logic_step(self, target: carla.Actor, actor_shapes, debug=False, sim_time=None,
                   shape_by_actor_id=None) -> SensorResult:
        """
        :param target:
        :param actor_shapes:
        """
        self.actor_sh_map.clear()
        self.actor_sh_map.load(actor_shapes)
        target_shape = self.actor_sh_map.get_shape_from_actor(target)

        # check the ghost probability
        num = random.random()
        ch = random.random()
        if ch < 0.5:
            if num <= self.sensor_probA.p_ghost:
                measurement = self._get_ghost_measurement(self.fov_polygon_A, target)
                return SensorResult(SensorResult.GHOST, measurement, self.sensor_probA, FovSector.REG_NAME_A, None)
        else:
            if num <= self.sensor_probB.p_ghost:
                measurement = self._get_ghost_measurement(self.fov_difference, target)
                return SensorResult(SensorResult.GHOST, measurement, self.sensor_probB, FovSector.REG_NAME_B, None)

        return SensorResult(SensorResult.NO_DETECTION, None, None, None, fov_region=None)
