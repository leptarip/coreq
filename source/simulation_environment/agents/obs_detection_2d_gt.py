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
import shapely
import shapely.ops
import math

import shapely.affinity
from shapely.validation import explain_validity
import source.simulation_environment.utils as utils
from source.simulation_environment.agents import traj_shape
from source.simulation_environment.agents.traj_shape import RoutePolygon
from source.simulation_environment.logs.printl import PrintL


class ObsDetection2D:
    TAG = "ObsDetection2D"

    DEF_LATERAL_MARGIN = 2
    DEF_EGO_DIM = (2, 4)  # width, length
    OBS_EPSILON_DISTANCE = 2

    def __init__(self,
                 ego_vehicle,
                 prediction_window,
                 ego_dim=DEF_EGO_DIM,
                 lateral_margin=DEF_LATERAL_MARGIN,
                 verbose_debug=False):
        self.ego_shape = None
        self.path_lx = None
        self.path_rx = None
        self.path_c = None
        self.path_poly = None
        self.distance_path = None
        self.fov_poly = None
        self.lateral_margin = lateral_margin
        self.ego_dim = ego_dim
        self.prediction_window = prediction_window
        self.ego = ego_vehicle
        self.obs_detected = False
        self.projected = None

        self.logger = PrintL(tag=ObsDetection2D.TAG, enabled=verbose_debug)

    def _get_dist_path_intersection(self, poly):
        # find the intersection (if any) between the path and the FOV boundary (FOV accounts for objects)
        if self.path_poly.is_valid:
            self.logger.to_print(message="is valid", header="{0}".format(self.ego.id), where=PrintL.STD_OUT)
        else:
            self.logger.to_print(message="is NOT valid, {0}".format(explain_validity(self.path_poly)),
                                 header="{0}".format(self.ego.id),
                                 where=PrintL.STD_OUT)
            return None

        intersection = shapely.intersection(self.path_poly, poly)

        if intersection is None or intersection.is_empty:
            self.logger.to_print(message="intersection is none", header="{0}".format(self.ego.id), where=PrintL.STD_OUT)
            return None
        if intersection.geom_type == 'MultiLineString':
            intersection = shapely.line_merge(intersection)

        shortest_line = shapely.shortest_line(self.ego_shape, intersection)

        if shortest_line is None:
            self.logger.to_print(message="shortest_line is none", where=PrintL.STD_OUT)
            return None

        # project the point to the path lines
        t = shortest_line.coords[-1]
        min_p = shapely.Point(t)
        shapely.prepare(min_p)

        d_c = shapely.line_locate_point(self.path_c, min_p)
        d_lx = shapely.line_locate_point(self.path_lx, min_p)
        d_rx = shapely.line_locate_point(self.path_rx, min_p)

        temp = [(d_c, self.path_c), (d_lx, self.path_lx), (d_rx, self.path_rx)]
        # res[0] contains the min distance, res[1] contains the associated path
        res = min(temp, key=lambda x: x[0])

        return shapely.ops.substring(res[1], start_dist=0, end_dist=res[0])

    def find_dist_path(self, proj_geoms):
        """
        # 1) find intersection between polygon path and all projected geometries
        # 2) if not 1), then find intersection between polygon path and polygon fov
        #   --in this case consider the border of polygon path and polygon fov
        # 3)
        #   A) check if the intersection geom intersects path_central, path_left or path_right
        #  -----
        #   B) if not, and the intersection is a polygon, get the closest point to the center of the car.
        #  ---- use find the projection of intersection geom on path_center, path_left, path_right
        #  --- take the shortest distance amongst the three projections.
        # 4) build the shortest path
        # --- take the path with the shortest projection and build the "red path line"
        :param proj_geoms:
        :return:
        """
        self.distance_path = self._get_dist_path_intersection(self.fov_poly.boundary)

        # distance_path == None means that there is no intersection
        # then, find the shortest path amongst central, left and right paths we have
        if self.distance_path is None:
            temp = [(self.path_c.length, self.path_c),
                    (self.path_lx.length, self.path_lx),
                    (self.path_rx.length, self.path_rx)]
            # res[0] contains the min distance, res[1] contains the associated path
            res = min(temp, key=lambda x: x[0])
            self.distance_path = res[1]

        if len(proj_geoms) > 0:
            proj_tree = shapely.STRtree(proj_geoms)
            indices = proj_tree.query(self.path_poly.boundary, predicate="intersects")
            if len(indices) > 0:
                list_of_paths = list()
                for geom in proj_tree.geometries.take(indices):
                    list_of_paths.append(self._get_dist_path_intersection(geom))

                # get the minimum path
                list_of_paths.append(self.distance_path)
                self.distance_path = min(list_of_paths, key=lambda x: shapely.length(x))

        if self.distance_path == self.path_c or self.distance_path == self.path_rx or self.distance_path == self.path_lx:
            self.obs_detected = False
        else:
            self.obs_detected = True

    def get_projection(self, actors: utils.ObstacleMap, target: traj_shape.Target):
        """
        This function should either return the projection of all the hit actors or change the
        return type as the target is only one!
        :param actors: all the actors
        :param target: the target actor to project the shape of
        :return: a list of one element or empty
        """
        self.projected = None
        if target is None:
            return list()
        if target.route_poly is None:
            return list()
        if not actors.is_item_hit(target.actor):
            return list()

        poly = None
        v = target.actor.get_velocity()
        vel_item_x = math.sqrt(v.x ** 2 + v.y ** 2)
        if math.fabs(vel_item_x) > 1:
            d = vel_item_x * self.prediction_window
            if d < target.route_poly.path_c.length and d < target.route_poly.path_lx.length and d < target.route_poly.path_rx.length:
                sub_c = shapely.ops.substring(target.route_poly.path_c, start_dist=0, end_dist=d)
                sub_lx = shapely.ops.substring(target.route_poly.path_lx, start_dist=0, end_dist=d)
                sub_rx = shapely.ops.substring(target.route_poly.path_rx, start_dist=0, end_dist=d)
                outer = [*sub_rx.coords,
                         sub_c.coords[-1],
                         *sub_lx.reverse().coords]

                poly = shapely.Polygon(outer)
                shapely.prepare(poly)
                self.projected = (target.actor, poly)
            else:
                self.projected = (target.actor, target.route_poly.polygon)
        if poly is None:
            return list()
        else:
            return [poly]

    @staticmethod
    def get_dist_to_target_in_fov(fov_polygon, ego_route:RoutePolygon, target: traj_shape.Target):
        dist = -1

        if ego_route is None or target is None or target.route_poly is None:
            return dist

        if ego_route.polygon is None or target.route_poly.actor_polygon is None:
            return dist

        if shapely.intersects(fov_polygon, target.route_poly.actor_polygon):
            if (shapely.intersects(ego_route.polygon, target.route_poly.actor_polygon) or
                    shapely.intersects(ego_route.actor_polygon, target.route_poly.polygon)) :
               dist = shapely.distance(ego_route.actor_polygon, target.route_poly.actor_polygon)

        return dist

    def step(self,
             fov_polygon: shapely.Polygon,
             route: RoutePolygon,
             actors: utils.ObstacleMap,
             target: traj_shape.Target,
             margin=0,
             debug=False) -> float:
        """
        Creates the geometry objects needed to compute the longitudinal distance to
        the next object
        :param target:
        :param fov_polygon:
        :param route:
        :param actors:
        :param margin:
        :param debug:
        :return: distance to the closest longitudinal obstacle
        """
        if route is None:
            self.projected = None
            return 0

        self.fov_poly = fov_polygon
        self.path_poly = route.polygon
        self.path_c = route.path_c
        self.path_lx = route.path_lx
        self.path_rx = route.path_rx
        self.ego_shape = route.actor_polygon

        projections = self.get_projection(actors, target)
        self.find_dist_path(projections)

        if self.distance_path is None:
            return 0

        le = shapely.length(self.distance_path)

        if debug:
            print("path to obs len " + str(le))
        return le
