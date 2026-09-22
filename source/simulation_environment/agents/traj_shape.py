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
import collections
from typing import Optional

import carla
import shapely
import source.simulation_environment.utils as utils

class RoutePolygon:
    def __init__(self,
                 actor_polygon: shapely.Polygon,
                 polygon: shapely.Polygon,
                 path_c: shapely.LineString,
                 path_rx: shapely.LineString,
                 path_lx: shapely.LineString):
        self.actor_polygon = actor_polygon
        self.polygon = polygon
        self.path_rx = path_rx
        self.path_lx = path_lx
        self.path_c = path_c

class Target:
    def __init__(self, actor: carla.Actor, actor_route: RoutePolygon):
        self.actor: carla.Actor = actor
        self.route_poly: RoutePolygon = actor_route

def get_route_polygon(actor: carla.Actor,
                      plan: collections.deque,
                      cache: Optional["RoutePolygonCache"] = None) -> Optional[RoutePolygon]:
    """
    @param actor: carla actor bounding box
    @param plan: the list of way points created by teh carla local planner with get_plan()
    @param cache: optional RoutePolygonCache. The plan is fixed after set_destination
        and only shrinks from the front, so the per-waypoint offsets need computing
        only once. Without a cache the original full rebuild is performed.
    @return: the polygon representing the route
    """
    if cache is not None:
        return cache.get(actor, plan)

    route_bb = list()
    route_bb_l = list()
    route_bb_r = list()
    route_c = list()
    extent_y = actor.bounding_box.extent.y
    r_ext = extent_y
    l_ext = -extent_y
    world_coords = utils.get_2d_box(actor)
    actor_polygon = shapely.Polygon((
        (world_coords[0, 0], world_coords[0, 1]),
        (world_coords[1, 0], world_coords[1, 1]),
        (world_coords[2, 0], world_coords[2, 1]),
        (world_coords[3, 0], world_coords[3, 1])
    ))

    shapely.prepare(actor_polygon)
    #r_vec = actor.get_transform().get_right_vector()
    #loc = actor.get_transform().location
    #p1 = loc + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
    #p2 = loc + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
    #route_bb_r.append([p1.x, p1.y])
    #route_bb_l.append([p2.x, p2.y])

    for wp, _ in plan:
        r_vec = wp.transform.get_right_vector()
        p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
        p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
        route_bb_r.append([p1.x, p1.y])
        route_bb_l.append([p2.x, p2.y])
        route_c.append([wp.transform.location.x, wp.transform.location.y])

    # Two points don't create a polygon, nothing to check
    if len(route_bb_r) < 2:
        return None

    line_rx = shapely.geometry.LineString(route_bb_r)
    line_c = shapely.geometry.LineString(route_c)
    line_lx = shapely.geometry.LineString(route_bb_l)

    route_bb.extend(route_bb_r)
    route_bb_l.reverse()
    route_bb.extend(route_bb_l)

    path_pol = shapely.Polygon(route_bb)
    shapely.prepare(path_pol)
    shapely.prepare(line_c)
    shapely.prepare(line_rx)
    shapely.prepare(line_lx)

    return RoutePolygon(actor_polygon, path_pol, line_c, line_rx, line_lx)


class RoutePolygonCache:
    """Caches the per-waypoint lateral offsets of a fixed route.

    The local planner is driven with ``stop_waypoint_creation=True``, so the plan is
    established once by ``set_destination`` and afterwards only ever loses entries
    from the front as the vehicle advances. The left/right/centre offsets are pure
    functions of immutable waypoints, so they are computed once and re-sliced from
    the current head instead of being rebuilt every tick.

    The offsets are produced with the same ``carla.Location`` arithmetic as
    ``get_route_polygon`` rather than recomputed in numpy: ``carla.Location`` stores
    float32, so evaluating the same expression in float64 yields different
    coordinates. Doing the identical arithmetic once keeps the result bit-exact.

    Only the route geometry is cached. ``actor_polygon`` depends on the current pose
    and is rebuilt on every call.
    """

    def __init__(self):
        self._plan_len = 0
        self._wp_ids = None
        self._route_r = None
        self._route_l = None
        self._route_c = None
        self._extent_y = None
        self._cached_head = None
        self._cached_geom = None

    def _precompute(self, actor, plan) -> None:
        extent_y = actor.bounding_box.extent.y
        r_ext = extent_y
        l_ext = -extent_y
        route_r, route_l, route_c, wp_ids = [], [], [], []

        for wp, _ in plan:
            r_vec = wp.transform.get_right_vector()
            p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_r.append([p1.x, p1.y])
            route_l.append([p2.x, p2.y])
            route_c.append([wp.transform.location.x, wp.transform.location.y])
            wp_ids.append(wp.id)

        self._plan_len = len(wp_ids)
        self._wp_ids = wp_ids
        self._route_r = route_r
        self._route_l = route_l
        self._route_c = route_c
        self._extent_y = extent_y
        self._cached_head = None
        self._cached_geom = None

    def _is_valid_window(self, actor, plan, head) -> bool:
        """The cached offsets describe this plan iff the window's ends still match."""
        if self._wp_ids is None or not 0 <= head < self._plan_len:
            return False
        if self._extent_y != actor.bounding_box.extent.y:
            return False
        return (self._wp_ids[head] == plan[0][0].id
                and self._wp_ids[-1] == plan[-1][0].id)

    def get(self, actor: carla.Actor, plan: collections.deque):
        # Two points don't create a polygon, nothing to check
        if len(plan) < 2:
            return None

        head = self._plan_len - len(plan)
        if not self._is_valid_window(actor, plan, head):
            self._precompute(actor, plan)
            head = 0

        if head != self._cached_head:
            route_bb_r = self._route_r[head:]
            route_bb_l = self._route_l[head:]
            route_c = self._route_c[head:]

            line_rx = shapely.geometry.LineString(route_bb_r)
            line_c = shapely.geometry.LineString(route_c)
            line_lx = shapely.geometry.LineString(route_bb_l)

            route_bb = list(route_bb_r)
            route_bb_l = list(route_bb_l)
            route_bb_l.reverse()
            route_bb.extend(route_bb_l)

            path_pol = shapely.Polygon(route_bb)
            shapely.prepare(path_pol)
            shapely.prepare(line_c)
            shapely.prepare(line_rx)
            shapely.prepare(line_lx)

            self._cached_head = head
            self._cached_geom = (path_pol, line_c, line_rx, line_lx)

        path_pol, line_c, line_rx, line_lx = self._cached_geom

        world_coords = utils.get_2d_box(actor)
        actor_polygon = shapely.Polygon((
            (world_coords[0, 0], world_coords[0, 1]),
            (world_coords[1, 0], world_coords[1, 1]),
            (world_coords[2, 0], world_coords[2, 1]),
            (world_coords[3, 0], world_coords[3, 1])
        ))
        shapely.prepare(actor_polygon)

        return RoutePolygon(actor_polygon, path_pol, line_c, line_rx, line_lx)


def show_path(debug, route: collections.deque):
    lt = -1
    for i in range(len(route)-1):

        debug.draw_line(
            route[i][0].transform.location + carla.Location(z=0.25),
            route[i+1][0].transform.location + carla.Location(z=0.25),
            thickness=0.5, color=carla.Color(0, 0, 0), life_time=lt, persistent_lines=False)
        i += 1

        debug.draw_line(
            route[-2][0].transform.location + carla.Location(z=0.25),
            route[-1][0].transform.location + carla.Location(z=0.25),
            thickness=0.5, color=carla.Color(0, 0, 0), life_time=lt, persistent_lines=False)
