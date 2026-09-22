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
import timeit
from multiprocessing import Pool

import shapely
import shapely.affinity
import math
import numpy as np
import source.simulation_environment.utils as utils
from source.simulation_environment.debug.timings import Timings


class Perception2Dgt:
    TAG = "[Perception2Dgt] "

    DEF_RAY_STEP = 1
    DEF_SENSOR_RANGE = 20
    MAX_SENSOR_FOV = 360
    WORKER_POOL_1_SIZE = 4
    OBS_EPSILON_DISTANCE = 2

    class Pair:
        def __init__(self, distance, index):
            self.distance = distance
            self.index = index

        @staticmethod
        def compare(item1, item2):
            if item1.distance < item2.distance:
                return -1
            elif item1.distance > item2.distance:
                return 1
            else:
                return 0

    def __init__(self,
                 buildings,
                 sensor,
                 fov=MAX_SENSOR_FOV,
                 sensor_range=DEF_SENSOR_RANGE,
                 ray_step=DEF_RAY_STEP
                 ):
        self.obstacle_map = utils.ObstacleMap()
        self.building_map = utils.ObstacleMap()
        self.fov_polygon = None
        self.blind_zones = None
        self.buildings = buildings
        self.sensor = sensor
        self.sensor_range = sensor_range
        self.ray_step = ray_step
        self.sTree_buildings = None
        self.sTree_obstacles = None
        self.fov = fov if fov < Perception2Dgt.MAX_SENSOR_FOV else Perception2Dgt.MAX_SENSOR_FOV
        self.visible_shapes = None
        self.pool = Pool(processes=Perception2Dgt.WORKER_POOL_1_SIZE)
        self.create_building_shapes()
        self.rays = self.test_init_fov_polygon()
        self.debug_time_test = Timings()
        self.debug_time_test_parallel = Timings()
        self.debug_time_fov_poly = Timings()

    def create_building_shapes(self):
        # it should be faster to use a local list than get all shapes from map as it runs again a for loop of same len
        building_shapes = list()
        for b in self.buildings:
            cords = utils.create_bb_points_2D(b.bounding_box)
            points = (
                (cords[0, 0], cords[0, 1]),
                (cords[1, 0], cords[1, 1]),
                (cords[2, 0], cords[2, 1]),
                (cords[3, 0], cords[3, 1])
            )
            polygon = shapely.Polygon(points)

            shapely.prepare(polygon)

            building_shapes.append(polygon)

            self.building_map.add_item(polygon, b)

        self.sTree_buildings = shapely.STRtree(building_shapes)

    def create_obstacle_shapes(self, obstacles):
        # it should be faster to use a local list than get all shapes from map as it runs again a for loop of same len
        obstacle_shapes = []

        for b in obstacles:
            world_coords = utils.create_bb_shape(b)
            polygon = shapely.Polygon((
                (world_coords[0, 0], world_coords[1, 0]),
                (world_coords[0, 1], world_coords[1, 1]),
                (world_coords[0, 2], world_coords[1, 2]),
                (world_coords[0, 3], world_coords[1, 3])
            ))

            shapely.prepare(polygon)
            obstacle_shapes.append(polygon)
            self.obstacle_map.add_item(polygon, b)

        self.sTree_obstacles = shapely.STRtree(obstacle_shapes)

    def create_fov_shape(self, points):
        """
       :param points: list of shapely points forming the fov polygon
       :return: the fov_polygon
       """
        self.fov_polygon = shapely.Polygon(tuple(points))

        shapely.prepare(self.fov_polygon)

        return self.fov_polygon

    def create_blind_fov_zones(self, points):
        """
       :param points: list of shapely points forming the ideal fov polygon
       :return: the fov_polygon
       """
        ideal_fov_poly = shapely.Polygon(tuple(points))
        shapely.prepare(ideal_fov_poly)
        self.blind_zones = ideal_fov_poly.difference(self.fov_polygon)
        shapely.prepare(self.blind_zones)

        return self.blind_zones

    def set_obstacles(self, list_of_obstacles):
        """
        create the shapes of the obstacles from the points
        list_of_obstacles = objects with attribute .bounding_box
        """
        self.obstacle_map.clear()
        self.obstacle_map.merge(self.building_map)
        self.create_obstacle_shapes(list_of_obstacles)


    def _get_start_angle(self):
        start = self.sensor.get_transform().rotation.yaw
        if self.fov < 360:
            angle = self.fov // 2
            start = start - angle
        return start

    def test_init_fov_polygon(self):

        start = self._get_start_angle()
        steps = self.fov // self.ray_step
        origin = shapely.Point(self.sensor.get_location().x, self.sensor.get_location().y)
        rays = list()

        for step in range(0, steps):
            angle = math.radians(start + self.ray_step * step)
            x = self.sensor_range * math.cos(angle)
            y = self.sensor_range * math.sin(angle)
            # creating the ray=(center, FOV exterior point)
            line = shapely.LineString([(0, 0), (x, y)])
            shapely.prepare(line)
            rays.append(line)

        return rays


    def test_compute_fov_polygon(self):
        # reposition the fov rays w.r.t the current position
        transform = self.sensor.get_transform()
        origin = shapely.Point(transform.location.x, transform.location.y)
        shapely.prepare(origin)
        rays = [shapely.affinity.translate(ray, xoff=transform.location.x, yoff=transform.location.y, zoff=0.0) for ray in self.rays]

        s_tree = shapely.STRtree(self.obstacle_map.get_all_shapes())
        hit_shapes = list()
        poly_points = list()
        ideal_poly_points = list()

        for ray in rays:
            indices = s_tree.query(ray, predicate="intersects")
            num_geoms = len(indices)
            p = ray.coords[1]
            if num_geoms == 0:
                poly_points.append(p)
                ideal_poly_points.append(p)
                continue
            elif num_geoms == 1:
                geom =  s_tree.geometries[indices[0]]
            else:
                s1 = shapely.STRtree(s_tree.geometries.take(indices))
                geom = s1.geometries.take(s1.nearest(origin))

            hit_shapes.append(geom)
            # find intersection
            inter = shapely.intersection(ray, geom)
            type_id = shapely.get_type_id(inter)
            if type_id == 1:
                # if the intersection is a line, take the first point
                p = shapely.Point(inter.coords[0])
            elif type_id == 0:
                # if the intersection is a point, take it
                p = inter

            poly_points.append(p)
            ideal_poly_points.append(ray.coords[1])

        if self.fov < Perception2Dgt.MAX_SENSOR_FOV:
            # add the central point to connect the FOV polygon to the center of the obps
            poly_points.append(origin)
            ideal_poly_points.append(origin)
        else:
            poly_points.append(poly_points[0])
            ideal_poly_points.append(ideal_poly_points[0])

        return poly_points, ideal_poly_points

    def test_poly_parallel(self):
        # reposition the fov rays w.r.t the current position
        transform = self.sensor.get_transform()
        origin = shapely.Point(transform.location.x, transform.location.y)
        shapely.prepare(origin)
        rays = [shapely.affinity.translate(ray, xoff=transform.location.x, yoff=transform.location.y, zoff=0.0) for ray
                in self.rays]

        s_tree = shapely.STRtree(self.obstacle_map.get_all_shapes())
        poly_points = list()
        ideal_poly_points = list()

        for result in self.pool.starmap(work_unit,
                                        ((ray, origin, s_tree) for ray in rays)
                                        ):
            poly_points.append(result[0])
            # save the ideal FOV polygon, the one if there were no intersections
            ideal_poly_points.append(result[1])

            if result[2] is not None:
                self.obstacle_map.set_hit(s_tree.geometries[result[2]], True)

        if self.fov < Perception2Dgt.MAX_SENSOR_FOV:
            # add the central point to connect the FOV polygon to the center of the obps
            poly_points.append(origin)
            ideal_poly_points.append(origin)
        else:
            poly_points.append(poly_points[0])
            ideal_poly_points.append(ideal_poly_points[0])

        return poly_points, ideal_poly_points

    def compute_fov_polygon(self):
        """
        :param self:
        :return: list of shapely points defining the fov polygon
        """
        origin = shapely.Point(self.sensor.get_location().x, self.sensor.get_location().y)

        s_tree = shapely.STRtree(self.obstacle_map.get_all_shapes())

        shapes = s_tree.geometries.take(
            s_tree.query(origin,
                        predicate="dwithin",
                        distance=self.sensor_range + Perception2Dgt.OBS_EPSILON_DISTANCE)
        ).tolist()

        s_tree = shapely.STRtree(shapes)

        poly_points = []
        ideal_poly_points = []

        # derive start and end points according to fov
        a_start = self._get_start_angle()

        steps = self.fov // self.ray_step

        for result in self.pool.starmap(_ray_cast,
                                        ((s_tree, origin, a_start, self.ray_step, step, self.sensor_range)
                                         for step in range(0, steps))
                                        ):
            poly_points.append(result[0])
            # save the ideal FOV polygon, the one if there were no intersections
            ideal_poly_points.append(result[1])

            for i in result[2]:
                self.obstacle_map.set_hit(shapes[i], True)

        if self.fov < Perception2Dgt.MAX_SENSOR_FOV:
            # add the central point to connect the FOV polygon to the center of the obps
            poly_points.append(origin)
            ideal_poly_points.append(origin)
        else:
            poly_points.append(poly_points[0])
            ideal_poly_points.append(ideal_poly_points[0])

        return poly_points, ideal_poly_points

    def set_visible_shapes(self):

        s_tree = shapely.STRtree(self.obstacle_map.get_hit_shapes())
        try:
            self.visible_shapes = s_tree.geometries.take(s_tree.query(self.fov_polygon, predicate="covers")).tolist()
            self.visible_shapes.extend(
                s_tree.geometries.take(s_tree.query(self.fov_polygon, predicate="intersects")).tolist())
        except Exception as e:
            print("Error visible shapes {0}", e)
            self.visible_shapes = []

    def step(self):
        """
        Creates the FoV polygon
        Creates the ideal Fov polygon, i.e., the one if there were no obstacles
        :return:
        """
        #time2 = timeit.default_timer()
        #fov_points, ideal_fov_points = self.compute_fov_polygon()
        #print("time create fov polygon: {0}".format(self.debug_time_fov_poly.get_new_avg(timeit.default_timer() - time2)))
        #self.create_fov_shape(fov_points)
        #self.create_blind_fov_zones(ideal_fov_points)
        #time2 = timeit.default_timer()
        #self.set_visible_shapes()
        #time2 = timeit.default_timer() - time2
        #print("time visible shapes: {0}".format(time2))
        time3 = timeit.default_timer()
        fov_points, ideal_fov_points = self.test_compute_fov_polygon()
        print("time test: {0}".format(self.debug_time_test.get_new_avg(timeit.default_timer() - time3)))
        self.create_fov_shape(fov_points)

        #time3 = timeit.default_timer()
        #a, b = self.test_poly_parallel()
        #print("time test parallel: {0}".format(self.debug_time_test_parallel.get_new_avg(timeit.default_timer() - time3)))

    def destroy(self):
        self.pool.close()

def work_unit(ray, origin, stree):
    indices = stree.query(ray, predicate="intersects")
    num_geoms = len(indices)
    p = ray.coords[1]
    inter_index = None
    if num_geoms == 0:
        return p, p, inter_index
    elif num_geoms == 1:
        inter_index = indices[0]
        geom = stree.geometries[inter_index]
    else:
        s1 = shapely.STRtree(stree.geometries.take(indices))
        inter_index = s1.nearest(origin)
        geom = s1.geometries[inter_index]

    # find intersection
    inter = shapely.intersection(ray, geom)
    type_id = shapely.get_type_id(inter)
    if type_id == 1:
        # if the intersection is a line, take the first point
        p = shapely.Point(inter.coords[0])
    elif type_id == 0:
        # if the intersection is a point, take it
        p = inter

    return p, ray.coords[1], inter_index

def _ray_cast(shape_tree, origin, a_start, ray_step, step, sensor_range):
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
    :param shape_tree:
    :param origin:
    :param a_start:
    :param ray_step:
    :param step:
    :param sensor_range:
    :return:
    """
    angle = math.radians(a_start + ray_step * step)
    x = sensor_range * math.cos(angle) + origin.x
    y = sensor_range * math.sin(angle) + origin.y
    # creating the ray=(center, FOV exterior point)
    line = shapely.LineString([(origin.x, origin.y), (x, y)])

    point_step = shapely.Point(x, y)

    # find all the intersections between the line=(center, ray) with the tree of obstacle shapes (objects)
    indices = shape_tree.query(line, predicate="intersects")
    hit_shapes_index = indices.tolist()
    geoms = shape_tree.geometries.take(indices).tolist()

    min_dist = sensor_range + 100  # big enough number
    min_point = p = point_step # store the shortest distance point

    for geom in geoms:
        inter = shapely.intersection(line, geom)
        type_id = shapely.get_type_id(inter)
        if type_id == 1:
            # if the intersection is a line, take the first point
            p = shapely.Point(inter.coords[0])
        elif type_id == 0:
            # if the intersection is a point, take it
            p = inter

        dist = origin.distance(p)
        if dist < min_dist:
            min_dist = dist
            min_point = p

    return min_point, point_step, hit_shapes_index
