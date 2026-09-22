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
import secrets
import string
import typing

import carla
import shapely
from typing import List, Tuple

import numpy as np
# import source.simulation_environment.matrix_helper as mh
import source.simulation_environment.cython.matrix_helper_cy as mh


class UtilPair:
    def __init__(self, key, value):
        self.key = key
        self.value = value

class ObstacleMap:
    class Item:
        def __init__(self, shape: shapely.Geometry, obstacle, hit=False):
            """

            :param shape: shapely.shape, 2D representation of obstacle
            :param obstacle: carla.Actor
            :param hit:
            """
            self.shape = shape
            self.obstacle = obstacle
            self.hit = hit
            self.prediction = None

        def get_obstacle_type(self) -> str:
            if isinstance(self.obstacle, carla.EnvironmentObject):
                return "building"
            return self.obstacle.type_id

    def __init__(self):
        self.map: typing.Dict[int, ObstacleMap.Item] = dict()

    def reset_hits(self):
        def reset(v):
            v.hit = False
            return v

        d = {k: reset(v) for k, v in self.map.items()}
        self.map = d

    def clear(self):
        self.map.clear()

    def merge(self, other):
        """
        Merges
        :param other:  utils.ObstacleMap
        :return:
        """
        self.map.update(other.map)

    def add_item(self, shape, obstacle, hit=False):
        """

        :param hit:
        :param shape: shapely shape
        :param obstacle: carla agent
        :return:
        """
        self.map[id(shape)] = ObstacleMap.Item(shape, obstacle, hit)

    def set_hit(self, shape, hit: bool):
        """

        :param hit:
        :param shape: shapely shape
        :return:
        """
        key = id(shape)
        if key in self.map:
            self.map[key].hit = hit

    def get_obstacle_type(self, shape) -> str:
        """

        :param shape:
        :return:
        """
        key = id(shape)
        if key in self.map:
            return self.map[key].get_obstacle_type()

        return ""

    def get_hit_obstacles(self):
        """
        :return:
        """

        def f(pair):
            key, value = pair
            return value.hit

        d = dict(filter(f, self.map.items()))
        return d

    def get_predicted_shapes(self):
        shapes = [item.prediction for item in self.map.values() if item.hit is True and item.prediction is not None]
        return shapes

    def get_hit_shapes(self):
        shapes = [item.shape for item in self.map.values() if item.hit is True]
        return shapes

    def get_all_shapes(self):
        shapes = [item.shape for item in self.map.values()]
        return shapes

    def get_item_by_shape(self, shape):
        return self.map[id(shape)]

    def is_item_hit(self, item):
        for value in self.map.values():
            if value.obstacle.id == item.id and value.hit:
                return True
        return False

    def get_item_by_actor_id(self, actor_id) -> Tuple[typing.Optional[int], typing.Optional[Item]]:
        for k, value in self.map.items():
            if value.obstacle.id == actor_id:
                return k, value
        return None, None




class ActorShapeMap:
    class Item:
        def __init__(self, actor, shape):
            self.actor = actor
            self.shape = shape
            self.visible = False

    def __init__(self):
        self.map_actor = {}
        self.map_shape = {}

    def clear(self):
        self.map_actor.clear()
        self.map_shape.clear()

    def load(self, actor_shape: List[UtilPair]):
        for pair in actor_shape:
            self.add(pair.key, pair.value)

    def add(self, actor, shape):
        item = ActorShapeMap.Item(actor, shape)
        self.map_actor[actor.id] = item
        self.map_shape[id(shape)] = item

    def set_actor_visible(self, actor):
        self.map_actor[actor.id].visible = True

    def set_shape_visible(self, shape):
        self.map_shape[id(shape)].visible = True

    def get_shape_from_actor(self, actor):
        return self.map_actor[actor.id].shape

    def is_shape_visible(self, actor):
        return self.map_actor[actor.id].visible

    def get_shapes(self):
        return [item.shape for item in self.map_shape.values()]


def sort_environment_objects(objects):
    """Return CARLA environment objects in a deterministic order.

    ``world.get_environment_objects()`` does not guarantee a stable order between
    server processes: Town05's 870 buildings came back in two different orders
    across six samples. That order propagates into ObstacleMap insertion order and
    from there into STRtree index order, which decides how ``nearest()`` breaks a
    tie between equidistant candidates. Pinning it keeps the perception input
    identical from run to run.

    :param objects: iterable of carla.EnvironmentObject
    :return: list ordered by bounding box pose, then extent, then id
    """
    def sort_key(obj):
        bounding_box = obj.bounding_box
        return (bounding_box.location.x,
                bounding_box.location.y,
                bounding_box.location.z,
                bounding_box.extent.x,
                bounding_box.extent.y,
                bounding_box.extent.z,
                obj.id)

    return sorted(objects, key=sort_key)


def key_or_default(dictionary, key, def_value):
    if key in dictionary:
        return dictionary[key]
    else:
        #print("[Warning]<Utils> Using default value {0} for not found key='{1}' in dictionary {2}"
        #      .format(def_value, key, dictionary))
        return def_value


def extend_3D_points(p: np.array):
    """
    Add 1s to a 3D point vector
    the input vector can contain multiple 3D vectors
    :param p: array of 3d points to extend to 4D by adding 1s
    :return:
    """
    try:
        a = np.empty(p.shape[1])
        a.fill(1)
        ex = np.vstack((p, a))
    except AttributeError:
        print("problem")
    return ex

def shapely_to_points(shapely_geom: shapely.Geometry):
    """
    # https://shapely.readthedocs.io/en/stable/reference/shapely.get_type_id.html#shapely.get_type_id
    # None(missing) is -1
    # POINT is 0
    # LINESTRING is 1
    # LINEARRING is 2
    # POLYGON is 3
    # MULTIPOINT is 4
    # MULTILINESTRING is 5
    # MULTIPOLYGON is 6
    # GEOMETRYCOLLECTION is 7
    :param shapely_geom:
    :return:
    """
    matrix = None
    shapely_type = shapely.get_type_id(shapely_geom)

    if shapely_type == 3:
        geom_coords = shapely_geom.exterior.xy
    elif shapely_type == 1:
        geom_coords = shapely_geom.xy
    else:
        print("[UTILS]<ERR> shapely_to_points error, type is {0}".format(shapely_type))
        return matrix
    matrix = np.zeros((3, len(geom_coords[0])))
    matrix[0, :] = np.expand_dims(np.array(geom_coords[0]), axis=0)
    matrix[1, :] = np.expand_dims(np.array(geom_coords[1]), axis=0)
    matrix[2, :] = 0

    return matrix

def create_bb_points_2D(bounding_box):
    """Return the 4 corners of a bounding box footprint as (4, 4) homogeneous rows.

    The box is *oriented*: CARLA reports a rotation alongside location and extent,
    and 727 of Town05's 870 building boxes carry a non-zero one. Taking
    ``location +/- extent`` on world axes, as this used to, yields the
    axis-aligned box of an oriented box -- a different footprint for 59% of them,
    with a median IoU against the true one of 0.64 and a worst case of 0.001.

    Only the yaw is applied. This is a ground-plane model, so pitch and roll would
    tilt the footprint out of the plane it lives in; no Town05 building box uses
    them. Near-zero sine/cosine terms are snapped, as shapely.affinity.rotate
    does, so the axis-aligned majority (yaw 0, +/-90, +/-180 -- 807 of 870 boxes)
    stays exact rather than picking up 1e-16 dirt.

    z is carried through unrotated; callers consume x and y only.
    """
    location = bounding_box.location
    extent = bounding_box.extent

    angle = np.radians(bounding_box.rotation.yaw)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    if abs(cos_a) < 2.5e-16:
        cos_a = 0.0
    if abs(sin_a) < 2.5e-16:
        sin_a = 0.0

    cords = np.zeros((4, 4))
    corners = ((extent.x, extent.y), (-extent.x, extent.y),
               (-extent.x, -extent.y), (extent.x, -extent.y))
    for row, (dx, dy) in enumerate(corners):
        cords[row, :] = np.array([location.x + dx * cos_a - dy * sin_a,
                                  location.y + dx * sin_a + dy * cos_a,
                                  location.z + extent.z, 1])
    return cords


def create_bb_shape(actor):
    """
    Takes a carla vehicle or class with bounding_box instance variable
    Returns the 2D bounds, 4 points.
    """
    b_coords = np.zeros((4, 4))
    extent = actor.bounding_box.extent

    b_coords[0, :] = np.array([extent.x,
                               extent.y,
                               extent.z, 1])
    b_coords[1, :] = np.array([-extent.x,
                               extent.y,
                               extent.z, 1])
    b_coords[2, :] = np.array([-extent.x,
                               -extent.y,
                               extent.z, 1])
    b_coords[3, :] = np.array([extent.x,
                               -extent.y,
                               extent.z, 1])

    bb_vehicle_matrix = mh.get_matrix_from_carla_location(actor.bounding_box.location)
    actor_world_matrix = mh.get_matrix_from_carla_transform(actor.get_transform())
    bb_world_matrix = np.dot(actor_world_matrix, bb_vehicle_matrix)
    world_coords = np.dot(bb_world_matrix, b_coords.T)
    return world_coords


def get_front_2d_box(actor) -> Tuple[float, float]:
    """
    Takes a carla vehicle or class with bounding_box instance variable
    Returns the 2D bounds, 4 points.
    """

    # extent = actor.bounding_box.extent
    # b_coords= np.array([extent.x, 0, extent.z, 1])
    #
    # bb_vehicle_matrix = mh.get_matrix_from_carla_location(actor.bounding_box.location)
    # actor_world_matrix = mh.get_matrix_from_carla_transform(actor.get_transform())
    # bb_world_matrix = np.dot(actor_world_matrix, bb_vehicle_matrix)
    # world_point = np.dot(bb_world_matrix, b_coords.T)
    trans = actor.get_transform()
    fw = trans.get_forward_vector()
    p1_x = trans.location.x + actor.bounding_box.extent.x * fw.x
    p1_y = trans.location.y + actor.bounding_box.extent.x * fw.y
    word_p = (p1_x, p1_y)
    return word_p


def get_2d_box_points(center, extent, fw, rg) \
        -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    return the points in counter clock-wise order (x,y) (x,-y) (-x,-y) (x,-y)
    :param center:
    :param extent: must have x and y attributes
    :param fw: forward vector with components x and y
    :param rg: right vector with components x and y
    :return:
    """

    ex_x_fw_x = extent.x * fw.x
    ex_x_fw_y = extent.x * fw.y

    front_x = center.x + ex_x_fw_x
    front_y = center.y + ex_x_fw_y
    back_x = center.x + (-1) * ex_x_fw_x
    back_y = center.y + (-1) * ex_x_fw_y

    ex_y_rg_x = extent.y * rg.x
    ex_y_rg_y = extent.y * rg.y

    # front right
    p1_x = front_x + ex_y_rg_x
    p1_y = front_y + ex_y_rg_y

    # front left
    p2_x = front_x - ex_y_rg_x
    p2_y = front_y - ex_y_rg_y

    # back left
    p3_x = back_x - ex_y_rg_x
    p3_y = back_y - ex_y_rg_y

    # back right
    p4_x = back_x + ex_y_rg_x
    p4_y = back_y + ex_y_rg_y

    return (p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y), (p4_x, p4_y)


def get_2d_box(actor):
    """
    Takes a carla vehicle or class with bounding_box instance variable
    Returns the 2D bounds, 4 points.
    """
    trans = actor.get_transform()
    fw = trans.get_forward_vector()
    rg = trans.get_right_vector()
    extent = actor.bounding_box.extent

    b_coords = np.zeros((2, 4))
    b_coords[0, :] = np.array([extent.x, extent.x, -extent.x, -extent.x])
    b_coords[1, :] = np.array([extent.y, -extent.y, -extent.y, extent.y])

    m = np.zeros((2, 2))
    m[0, :] = np.array([fw.x, rg.x])
    m[1, :] = np.array([fw.y, rg.y])

    mult = np.dot(m, b_coords)

    world_coords = np.zeros((2, 4))
    world_coords[0, :] = mult[0, :] + trans.location.x
    world_coords[1, :] = mult[1, :] + trans.location.y

    return world_coords.T


def get_2d_box_parametrized(center, extent, fw, rg):
    """
    Takes a carla vehicle or class with bounding_box instance variable
    Returns the 2D bounds, 4 points.
    """

    b_coords = np.zeros((2, 4))
    b_coords[0, :] = np.array([extent.x, extent.x, -extent.x, -extent.x])
    b_coords[1, :] = np.array([extent.y, -extent.y, -extent.y, extent.y])

    m = np.zeros((2, 2))
    m[0, :] = np.array([fw.x, rg.x])
    m[1, :] = np.array([fw.y, rg.y])

    mult = np.dot(m, b_coords)

    world_coords = np.zeros((2, 4))
    world_coords[0, :] = mult[0, :] + center.x
    world_coords[1, :] = mult[1, :] + center.y

    return world_coords.T


def get_random_point_in_poly(poly, attempts=100):
    point = poly.point_on_surface()
    minx, miny, maxx, maxy = poly.bounds

    for i in range(attempts):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)

        if shapely.contains_xy(poly, x=x, y=y):
            point = shapely.Point(x, y)
            break

    return point

def are_shapes_colliding(shape_1, shape_2) -> bool:
    return shape_1.covers(shape_2) or shape_1.intersects(shape_2)


def create_experiment_name():
    # Worker processes may be forked from the same parent and therefore inherit
    # identical ``random`` state.  Redis experiment keys are operational IDs,
    # not scenario randomness, so use OS entropy to prevent cross-worker log
    # corruption after synchronized CARLA restarts.
    h_set = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(12))
    h_set = h_set.replace(":", "p")
    return h_set


def create_sensor_msg_key(key: str, sensor_name: str):
    value = key + ":" + sensor_name
    return value


def create_sensor_name(sensor_name: str):
    c = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    c = c.replace(":", "t")
    n = sensor_name + "_" + c
    return n
