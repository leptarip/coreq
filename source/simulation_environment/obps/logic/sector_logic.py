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
from typing import Optional

import carla
import shapely
import math

import shapely.affinity
from source.simulation_environment.cfg.sensor_cfg import SensorCfg, Fov, sample_uncertainty
import source.simulation_environment.utils as utils
from source.simulation_environment.obps.measurement import Measurement, MeasuredPoint, MeasuredValue, Measured2dBox
from source.simulation_environment.obps.sensor_prob import SensorProb
from source.simulation_environment.obps.sensor_result import SensorResult
from source.simulation_environment.rare_events.wrapper import (
    ImportanceSampler,
    require_sampler_for_bias,
)
from source.simulation_environment.rare_events.window import (
    bias_window_active,
    bias_window_contains_time,
    parse_bias_window,
)


class FovSector(Fov):
    SCALE = 0.75
    RADIUS = 50
    PROB_DET = 1
    PROB_MISS = 0
    PROB_GHOST = 0
    WRONG = 0
    REG_NAME_A = "A"
    REG_NAME_B = "B"
    UNC_TYPE_NORMAL = "normal"
    UNC_TYPE_WORSE_CASE = "worst_case"

    def __init__(self, cfg: dict, position, fov_type=SensorCfg.TYPE_SECTOR):
        super().__init__(fov_type, position, cfg["orientation"])

        conf = cfg["fov"]["sector"]

        self.fov_degrees = conf["degrees"]
        self.radius = utils.key_or_default(conf, "radius", FovSector.RADIUS)
        self.scale = utils.key_or_default(conf, "scale", FovSector.SCALE)
        self.probA_det = utils.key_or_default(conf["probA"], "det", FovSector.PROB_DET)
        self.probA_miss = utils.key_or_default(conf["probA"], "miss", FovSector.PROB_MISS)
        self.probA_ghost = utils.key_or_default(conf["probA"], "ghost", FovSector.PROB_GHOST)
        self.probA_wrong = utils.key_or_default(conf["probA"], "wrong", FovSector.WRONG)
        self.probB_det = utils.key_or_default(conf["probB"], "det", FovSector.PROB_DET)
        self.probB_miss = utils.key_or_default(conf["probB"], "miss", FovSector.PROB_MISS)
        self.probB_ghost = utils.key_or_default(conf["probB"], "ghost", FovSector.PROB_GHOST)
        self.probB_wrong = utils.key_or_default(conf["probB"], "wrong", FovSector.WRONG)
        self.unc_pos = utils.key_or_default(conf["uncertainty"], "pos", 0)
        self.unc_vel = utils.key_or_default(conf["uncertainty"], "vel", 0)
        self.unc_type = utils.key_or_default(conf["uncertainty"], "type", FovSector.UNC_TYPE_NORMAL)


class SectorLogic:
    TAG = "[SectorLogic] "
    MAX_SENSOR_FOV = 360

    def __init__(self, cfg: FovSector, sensor_bias: Optional[dict] = None, sampler: Optional[ImportanceSampler] = None,
                 seed: Optional[int] = None):

        require_sampler_for_bias(sampler, sensor_bias, "Sensor")
        self.target_shape = None
        self.target_in_A = False
        self.target_in_B = False
        self.cfg = cfg
        self.sensor_bias = sensor_bias or {}
        self.bias_window = parse_bias_window(self.sensor_bias)
        self._window_successes = 0
        self._proposal_active = True
        self.sampler = sampler
        self.seed = seed
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

        # Precompute normalized base/bias weights for A/B regions
        self.miss_scale = float(self.sensor_bias.get("miss_scale", 1.0))
        self.bias_unc_pos_add = float(self.sensor_bias.get("unc_p_add", 0.0))
        self.bias_unc_vel_add = float(self.sensor_bias.get("unc_v_add", 0.0))

        self.weights_A_base = self._normalize_weights(self.weights_A)
        self.weights_B_base = self._normalize_weights(self.weights_B)
        self.weights_A_bias = self._build_bias_weights(self.weights_A_base)
        self.weights_B_bias = self._build_bias_weights(self.weights_B_base)
        audit_configs = self.sensor_bias.get("audit_proposals", {}) or {}
        audit_scales = self.sensor_bias.get("audit_miss_scales", {}) or {}
        if audit_configs and audit_scales:
            raise ValueError(
                "Use either audit_proposals or the legacy audit_miss_scales, not both."
            )
        if not audit_configs:
            audit_configs = {
                str(proposal_id): {
                    "miss_scale": float(scale),
                    **(
                        {"active_window": self.sensor_bias["active_window"]}
                        if "active_window" in self.sensor_bias else {}
                    ),
                }
                for proposal_id, scale in audit_scales.items()
            }
        self.audit_sensor_biases = {
            str(proposal_id): dict(values)
            for proposal_id, values in audit_configs.items()
        }
        sampler_audit_ids = set(
            getattr(self.sampler, "audit_proposal_ids", ())
        ) if self.sampler else set()
        if self.sampler and set(self.audit_sensor_biases) != sampler_audit_ids:
            raise ValueError(
                "Sensor audit proposals must match the sampler audit proposal IDs."
            )
        self.audit_weights_A_bias = {
            proposal_id: self._build_bias_weights_for_scale(
                self.weights_A_base,
                float(values.get("miss_scale", 1.0)),
            )
            for proposal_id, values in self.audit_sensor_biases.items()
        }
        self.audit_weights_B_bias = {
            proposal_id: self._build_bias_weights_for_scale(
                self.weights_B_base,
                float(values.get("miss_scale", 1.0)),
            )
            for proposal_id, values in self.audit_sensor_biases.items()
        }
        self.audit_bias_windows = {
            proposal_id: parse_bias_window(values)
            for proposal_id, values in self.audit_sensor_biases.items()
        }
        self._audit_window_successes = {
            proposal_id: 0 for proposal_id in self.audit_sensor_biases
        }
        if self.seed is not None:
            random.seed(self.seed)

    @staticmethod
    def _normalize_weights(weights):
        vals = [max(float(w), 0.0) for w in weights]
        total = sum(vals)
        if total <= 0:
            return [1.0 / len(vals)] * len(vals)
        return [w / total for w in vals]

    def _build_bias_weights(self, base_weights):
        return self._build_bias_weights_for_scale(base_weights, self.miss_scale)

    @classmethod
    def _build_bias_weights_for_scale(cls, base_weights, miss_scale):
        w_det, w_wrong, w_miss = base_weights
        w_miss *= float(miss_scale)
        return cls._normalize_weights([w_det, w_wrong, w_miss])

    def _audit_probabilities(self, base_weights, audit_weights, sim_time):
        return {
            proposal_id: (
                weights
                if bias_window_active(
                    self.audit_bias_windows[proposal_id], sim_time,
                    self._audit_window_successes[proposal_id],
                )
                else base_weights
            )
            for proposal_id, weights in audit_weights.items()
        }

    def _record_window_success(self, poll, sim_time):
        if poll == SensorResult.MISS:
            return
        if bias_window_contains_time(self.bias_window, sim_time):
            self._window_successes += 1
        for proposal_id, window in self.audit_bias_windows.items():
            if bias_window_contains_time(window, sim_time):
                self._audit_window_successes[proposal_id] += 1

    def compute_unc_v(self) -> float:
        return sample_uncertainty(
            self.cfg.unc_vel,
            self.bias_unc_vel_add if self._proposal_active else 0.0,
            self.cfg.unc_type,
            self.sampler,
            component=(
                "sensor.uncertainty.velocity.window"
                if self.bias_window is not None and self._proposal_active
                else "sensor.uncertainty.velocity"
            ),
        )

    def compute_unc_p(self) -> float:
        return sample_uncertainty(
            self.cfg.unc_pos,
            self.bias_unc_pos_add if self._proposal_active else 0.0,
            self.cfg.unc_type,
            self.sampler,
            component=(
                "sensor.uncertainty.position.window"
                if self.bias_window is not None and self._proposal_active
                else "sensor.uncertainty.position"
            ),
        )

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

    def _get_detection_result(self, target, target_shape) -> Measurement:
        global_carla_id = target.id
        v = target.get_velocity()
        vel = math.sqrt(v.x ** 2 + v.y ** 2)
        transform = target.get_transform()
        front_x, front_y = utils.get_front_2d_box(target)
        unc_p = 0
        unc_v = 0

        if self.cfg.unc_pos != 0:
            unc_p = self.compute_unc_p()
            box_2d = shapely.affinity.translate(target_shape, xoff=unc_p, yoff=unc_p)
        else:
            box_2d = target_shape
        x = transform.location.x + unc_p
        y = transform.location.y + unc_p
        front_x = front_x + unc_p
        front_y = front_y + unc_p

        if self.cfg.unc_vel != 0:
            unc_v = self.compute_unc_v()
        vel = vel + unc_v

        return Measurement(
            target_id=global_carla_id,
            position=MeasuredPoint(x, y, uncertainty=self.cfg.unc_pos),
            front=MeasuredPoint(front_x, front_y, uncertainty=self.cfg.unc_pos),
            v=MeasuredValue(vel, uncertainty=self.cfg.unc_vel),
            length=MeasuredValue(target.bounding_box.extent.x),
            width=MeasuredValue(target.bounding_box.extent.y),
            box2d=Measured2dBox(box_2d.exterior.coords[0],
                                box_2d.exterior.coords[1],
                                box_2d.exterior.coords[2],
                                box_2d.exterior.coords[3],
                                uncertainty=self.cfg.unc_pos),
            semantic_tag=MeasuredValue("vehicle")
        )

    def _get_ghost_measurement(self, fov, target) -> Measurement:
        global_carla_id = target.id
        point = utils.get_random_point_in_poly(fov)
        v = target.get_velocity()
        vel = math.sqrt(v.x ** 2 + v.y ** 2)
        fw = target.get_transform().get_forward_vector()
        front_x = point.x + target.bounding_box.extent.x * fw.x
        front_y = point.y + target.bounding_box.extent.x * fw.y
        p1, p2, p3, p4 = utils.get_2d_box_points(point,
                                                 target.bounding_box.extent,
                                                 fw,
                                                 target.get_transform().get_right_vector())

        if self.cfg.unc_vel != 0:
            unc = self.compute_unc_v()
            vel = vel + unc

        return Measurement(
            target_id=global_carla_id,
            position=MeasuredPoint(point.x, point.y, uncertainty=self.cfg.unc_pos),
            front=MeasuredPoint(front_x, front_y, uncertainty=self.cfg.unc_pos),
            v=MeasuredValue(vel, uncertainty=self.cfg.unc_vel),
            length=None,
            width=None,
            box2d=Measured2dBox(p1, p2, p3, p4, uncertainty=self.cfg.unc_pos)
        )

    def _get_wrong_measurement(self, target, target_shape) -> Measurement:
        # TODO implement the WRONG function!
        return self._get_detection_result(target, target_shape)

    def compute_result_A(self, poll, target, target_shape) -> SensorResult:
        """

        :param poll:
        :param target:
        :param target_shape:
        :return:
        """
        if poll == SensorResult.DETECTION:
            measurement = self._get_detection_result(target, target_shape)
            return SensorResult(poll, measurement, self.sensor_probA, FovSector.REG_NAME_A, fov_region=None)

        elif poll == SensorResult.WRONG:
            measurement = self._get_wrong_measurement(target, target_shape)
            return SensorResult(poll, measurement, self.sensor_probA, FovSector.REG_NAME_A, fov_region=None)

        elif poll == SensorResult.MISS:
            return SensorResult(poll, None, self.sensor_probA, FovSector.REG_NAME_A, fov_region=None)

        return SensorResult(poll, None, None, None, fov_region=None)

    def compute_result_B(self, poll, target, target_shape) -> SensorResult:
        """

        :param poll:
        :param target:
        :param target_shape:
        :return:
        """
        if poll == SensorResult.DETECTION:
            measurement = self._get_detection_result(target, target_shape)
            return SensorResult(poll, measurement, self.sensor_probB, FovSector.REG_NAME_B, fov_region=None)

        elif poll == SensorResult.WRONG:
            measurement = self._get_wrong_measurement(target, target_shape)
            return SensorResult(poll, measurement, self.sensor_probB, FovSector.REG_NAME_B, fov_region=None)

        elif poll == SensorResult.MISS:
            return SensorResult(poll, None, self.sensor_probB, FovSector.REG_NAME_B, fov_region=None)

        return SensorResult(poll, measurement=None, sensor_prob=None, fov_region_name=None, fov_region=None)

    def logic_step(self, target: carla.Actor, actor_shapes, debug=False, sim_time=None,
                   shape_by_actor_id=None) -> SensorResult:
        """
        :param target:
        :param actor_shapes:
        :param debug:
        :param shape_by_actor_id: actor id -> shape, built once per tick by ObpsManager.
            When absent, fall back to rebuilding the local map (same result, slower).
        """
        if shape_by_actor_id is not None:
            target_shape = shape_by_actor_id[target.id]
        else:
            self.actor_sh_map.clear()
            self.actor_sh_map.load(actor_shapes)
            target_shape = self.actor_sh_map.get_shape_from_actor(target)
        self.target_in_A = self.fov_polygon_A.covers(target_shape) or self.fov_polygon_A.intersects(target_shape)
        self.target_in_B = self.fov_polygon_B.covers(target_shape) or self.fov_polygon_B.intersects(target_shape)
        self._proposal_active = bias_window_active(
            self.bias_window, sim_time, self._window_successes
        )
        component = (
            "sensor.outcome.window"
            if self.bias_window is not None and self._proposal_active
            else "sensor.outcome"
        )

        if self.target_in_A:
            if self.sampler:
                idx = self.sampler.categorical(
                    self.weights_A_base,
                    self.weights_A_bias if self._proposal_active else self.weights_A_base,
                    component=component,
                    audit_probs_bias=self._audit_probabilities(
                        self.weights_A_base, self.audit_weights_A_bias, sim_time
                    ),
                )
                poll = self.choices[idx]
            else:
                poll = random.choices(self.choices, weights=self.weights_A_base, k=1)[0]
            res = self.compute_result_A(poll, target, target_shape)
            self._record_window_success(poll, sim_time)
            return res
        elif self.target_in_B:
            if self.sampler:
                idx = self.sampler.categorical(
                    self.weights_B_base,
                    self.weights_B_bias if self._proposal_active else self.weights_B_base,
                    component=component,
                    audit_probs_bias=self._audit_probabilities(
                        self.weights_B_base, self.audit_weights_B_bias, sim_time
                    ),
                )
                poll = self.choices[idx]
            else:
                poll = random.choices(self.choices, weights=self.weights_B_base, k=1)[0]
            res = self.compute_result_B(poll, target, target_shape)
            self._record_window_success(poll, sim_time)
            return res
        else:
            # check the ghost probability
            choose_region_A = (
                self.sampler.bernoulli(
                    p_base=0.5, p_bias=0.5, component="sensor.ghost_region"
                )
                if self.sampler else random.random() < 0.5
            )
            if choose_region_A:
                trigger = (self.sampler.bernoulli(
                               p_base=self.sensor_probA.p_ghost,
                               p_bias=self.sensor_probA.p_ghost,
                               component="sensor.ghost_trigger")
                           if self.sampler else random.random() <= self.sensor_probA.p_ghost)
                if trigger:
                    measurement = self._get_ghost_measurement(self.fov_polygon_A, target)
                    return SensorResult(SensorResult.GHOST, measurement, self.sensor_probA, FovSector.REG_NAME_A, None)
            else:
                trigger = (self.sampler.bernoulli(
                               p_base=self.sensor_probB.p_ghost,
                               p_bias=self.sensor_probB.p_ghost,
                               component="sensor.ghost_trigger")
                           if self.sampler else random.random() <= self.sensor_probB.p_ghost)
                if trigger:
                    measurement = self._get_ghost_measurement(self.fov_difference, target)
                    return SensorResult(SensorResult.GHOST, measurement, self.sensor_probB, FovSector.REG_NAME_B, None)

        return SensorResult(SensorResult.NO_DETECTION, None, None, fov_region_name=None, fov_region=None)
