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

import numpy as np
import shapely
import shapely.affinity
import math
import source.simulation_environment.utils as utils
from source.simulation_environment.debug.timings import Timings


class Perception2Dgt:
    TAG = "[Perception2Dgt] "

    DEF_SENSOR_RESOLUTION = 1 # angle in degrees between two cast rays
    DEF_SENSOR_RANGE = 70 # in [m]
    MAX_SENSOR_FOV = 360 # angle in degrees
    DEFAULT_SENSOR_FOV = 270  # angle in degrees

    def __init__(self,
                 buildings,
                 sensor,
                 fov=MAX_SENSOR_FOV,
                 sensor_range=DEF_SENSOR_RANGE,
                 ray_step=DEF_SENSOR_RESOLUTION
                 ):
        self.actors_map = utils.ObstacleMap() # contains all carla actors rather than ego
        self.obstacle_map = utils.ObstacleMap()
        self.building_map = utils.ObstacleMap()
        self.fov_polygon = None
        self.blind_zones = None
        self.buildings = buildings
        self.sensor = sensor
        self.sensor_range = sensor_range
        self.ray_step = ray_step
        self.fov = fov if fov < Perception2Dgt.MAX_SENSOR_FOV else Perception2Dgt.MAX_SENSOR_FOV
        # telemetry for the tie guard in compute_fov_polygon
        self.tie_groups = 0
        self.tie_fallbacks = 0
        self.create_building_shapes()
        self.rays = self.init_fov_polygon()
        self.rays_mls = self._init_rays_multilinestring()
        self.debug_time_test = Timings()


    def create_building_shapes(self):
        """Build the 2D footprint of every static obstacle, once, keeping one
        polygon per distinct footprint.

        This perception model is purely 2D: create_bb_points_2D takes the x/y of
        the bounding box and discards z entirely. CARLA reports a multi-storey
        building as one EnvironmentObject *per floor*, all sharing the same x/y
        box and differing only in bounding_box.location.z, so every floor of a
        tower collapses onto exactly the same footprint. On Town05 that is 870
        objects for 486 distinct footprints -- 44% redundant, with groups running
        up to 20 deep.

        Duplicates are not free: they enter the STRtree, are returned as extra
        candidates by every ray query that hits the building, and make ties for
        the nearest candidate the norm rather than the exception. Keeping one
        polygon per distinct footprint changes nothing geometrically -- the
        discarded copies are identical -- while shrinking the tree and the
        candidate lists.

        The key is the exact corner tuple, so only byte-identical footprints
        collapse. Two objects describing the same rectangle with a different
        corner order are both kept, which is conservative but safe.
        """
        # it should be faster to use a local list than get all shapes from map as it runs again a for loop of same len
        building_shapes = list()
        seen_footprints = set()
        for b in self.buildings:
            cords = utils.create_bb_points_2D(b.bounding_box)
            points = (
                (cords[0, 0], cords[0, 1]),
                (cords[1, 0], cords[1, 1]),
                (cords[2, 0], cords[2, 1]),
                (cords[3, 0], cords[3, 1])
            )
            if points in seen_footprints:
                # another floor of a building already represented by this footprint
                continue
            seen_footprints.add(points)

            polygon = shapely.Polygon(points)

            shapely.prepare(polygon)

            building_shapes.append(polygon)

            self.building_map.add_item(polygon, b)

    def create_actor_shapes(self, obstacles):
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
            self.actors_map.add_item(polygon, b)

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
        self.actors_map.clear()
        self.create_actor_shapes(list_of_obstacles)
        self.obstacle_map.clear()
        self.obstacle_map.merge(self.actors_map)
        self.obstacle_map.merge(self.building_map)



    def _get_start_angle(self):
        start = self.sensor.get_transform().rotation.yaw
        if self.fov < 360:
            angle = self.fov // 2
            start = start - angle
        return start

    def init_fov_polygon(self):

        start = self._get_start_angle()
        steps = self.fov // self.ray_step
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

    def _init_rays_multilinestring(self):
        """Pack the base rays into one geometry so the per-tick pose update is a
        single affine transform instead of one shapely call per ray.

        shapely applies an affine transform to a geometry's whole coordinate array
        in one vectorized pass, so transforming the collection performs exactly the
        same arithmetic, in the same order, as transforming each ray individually.
        """
        return shapely.MultiLineString([list(ray.coords) for ray in self.rays])


    def compute_fov_polygon(self):
        """
        # https://shapely.readthedocs.io/en/stable/reference/shapely.get_type_id.html#shapely.get_type_id
        # None(missing) is -1
        # POINT is 0
        # LINESTRING is 1
        """
        # reposition the fov rays w.r.t the current position
        transform = self.sensor.get_transform()
        origin = shapely.Point(transform.location.x, transform.location.y)
        shapely.prepare(origin)
        # Transform all rays with two affine calls on the packed collection rather
        # than two per ray. Same arithmetic, same order, ~25x fewer shapely calls.
        rays_geom = shapely.affinity.rotate(self.rays_mls, transform.rotation.yaw, origin=(0, 0), use_radians=False)
        rays_geom = shapely.affinity.translate(rays_geom, xoff=transform.location.x, yoff=transform.location.y, zoff=0.0)
        rays = shapely.get_parts(rays_geom)
        # endpoint of every 2-point ray, in one call
        ray_ends = shapely.get_coordinates(rays_geom)[1::2]

        # create the stree of all obstacles, including all carla actors plus buildings
        s_tree = shapely.STRtree(self.obstacle_map.get_all_shapes())

        # TRIAL: whole-fan vectorisation. Every geometric operation still runs in
        # GEOS (distance / intersection / get_type_id are ufuncs); numpy only does
        # the bookkeeping, i.e. a segmented argmin to pick the nearest candidate
        # per ray in place of a throwaway STRtree per ray.
        hit_ray_idx, hit_geom_idx = s_tree.query(rays, predicate="intersects")
        poly_xy = ray_ends.copy()

        if hit_ray_idx.size:
            hit_rays, starts = np.unique(hit_ray_idx, return_index=True)
            counts = np.diff(np.append(starts, hit_ray_idx.size))
            candidates = s_tree.geometries.take(hit_geom_idx)

            # nearest candidate per ray: first minimum within each contiguous group,
            # which is what STRtree.nearest returns when the minimum is unique
            distances = shapely.distance(origin, candidates)
            group_min = np.repeat(np.minimum.reduceat(distances, starts), counts)
            rel = np.arange(distances.size) - np.repeat(starts, counts)
            sentinel = distances.size + 1
            first_rel = np.minimum.reduceat(np.where(distances == group_min, rel, sentinel), starts)
            chosen = candidates[starts + first_rel]

            # Guard. argmin and STRtree.nearest can only disagree when a ray's
            # minimum distance is not unique. Where that happens, the pick is
            # immaterial if the tied candidates are the same geometry -- which is
            # the usual case, because CARLA reports many building parts sharing one
            # bounding box. If they are NOT identical, fall back to the exact
            # per-ray nearest() so the vertex matches the scalar implementation by
            # construction rather than by observation.
            is_min = distances == group_min
            tie_count = np.add.reduceat(is_min.astype(np.intp), starts)
            ambiguous = tie_count > 1
            if ambiguous.any():
                self.tie_groups += int(ambiguous.sum())
                group_of = np.repeat(np.arange(starts.size), counts)
                tied = is_min & ambiguous[group_of]
                identical = shapely.equals_exact(candidates[tied], chosen[group_of[tied]], 0.0)
                if not identical.all():
                    for group in np.unique(group_of[tied][~identical]):
                        span = candidates[starts[group]:starts[group] + counts[group]]
                        sub_tree = shapely.STRtree(span)
                        chosen[group] = sub_tree.geometries.take(sub_tree.nearest(origin))
                        self.tie_fallbacks += 1

            for geom in chosen:
                self.obstacle_map.set_hit(geom, True)
                self.actors_map.set_hit(geom, True)

            inter = shapely.intersection(rays[hit_rays], chosen)
            type_id = shapely.get_type_id(inter)
            coords, coord_index = shapely.get_coordinates(inter, return_index=True)
            per_geom = np.bincount(coord_index, minlength=inter.size)
            first = np.zeros(inter.size, dtype=np.intp)
            first[1:] = np.cumsum(per_geom)[:-1]
            # only POINT / LINESTRING contribute a vertex, matching the loop's branches
            keep = ((type_id == 0) | (type_id == 1)) & (per_geom > 0)
            poly_xy[hit_rays[keep]] = coords[first[keep]]

        poly_points = [tuple(xy) for xy in poly_xy]
        ideal_poly_points = [tuple(xy) for xy in ray_ends]

        if self.fov < self.MAX_SENSOR_FOV:
            # add the central point to connect the FOV polygon to the center of the obps
            poly_points.append(origin)
            ideal_poly_points.append(origin)
        else:
            poly_points.append(poly_points[0])
            ideal_poly_points.append(ideal_poly_points[0])

        return poly_points, ideal_poly_points


    @property
    def visible_shapes(self):
        """Obstacles inside the current FOV.

        Derived state, computed when read rather than at the end of every step().
        The only consumer is the debug viewer, so a headless run used to build
        this list on every agent-tick and never look at it. Measured saving is
        small (~0.15 s per intersection2 episode, nothing measurable on
        intersection1) -- the reason to do it is that the work is dead in the
        tick loop, not the wall clock.

        Computed at read time, so the debug viewer sees the FOV and hit flags as
        they stand when it asks rather than as they stood at the end of step().
        It reads immediately after stepping, so this is equivalent in practice.

        Returns None before the first step, matching the attribute it replaces.
        """
        if self.fov_polygon is None:
            return None

        s_tree = shapely.STRtree(self.obstacle_map.get_hit_shapes())
        try:
            shapes = s_tree.geometries.take(s_tree.query(self.fov_polygon, predicate="covers")).tolist()
            shapes.extend(
                s_tree.geometries.take(s_tree.query(self.fov_polygon, predicate="intersects")).tolist())
            return shapes
        except Exception as e:
            print("Error visible shapes {0}", e)
            return []

    def step(self):
        """
        Creates the FoV polygon
        Creates the ideal Fov polygon, i.e., the one if there were no obstacles
        :return:
        """
        time3 = timeit.default_timer()
        fov_points, ideal_fov_points = self.compute_fov_polygon()
        #print("[PERCEPTION_2D] time test: {0}".format(self.debug_time_test.get_new_avg(timeit.default_timer() - time3)))
        self.create_fov_shape(fov_points)

