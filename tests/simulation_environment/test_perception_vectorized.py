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

from types import SimpleNamespace

import numpy as np
import pytest
import shapely
import shapely.affinity

from source.simulation_environment.agents.perception_2d_gt import Perception2Dgt


class _Sensor:
    def __init__(self):
        self.transform = _transform(0.0, 0.0, 0.0)

    def get_transform(self):
        return self.transform


class _Actor:
    def __init__(self, actor_id):
        self.id = actor_id
        self.type_id = "vehicle.test"


def _transform(x, y, yaw):
    return SimpleNamespace(
        location=SimpleNamespace(x=x, y=y),
        rotation=SimpleNamespace(yaw=yaw),
    )


def _clone(geometry):
    return shapely.from_wkb(shapely.to_wkb(geometry))


def _make_perception(geometries, *, fov=360, ray_step=5, sensor_range=70):
    sensor = _Sensor()
    perception = Perception2Dgt(
        [], sensor, fov=fov, ray_step=ray_step, sensor_range=sensor_range
    )
    for actor_id, geometry in enumerate(geometries):
        perception.actors_map.add_item(_clone(geometry), _Actor(actor_id))
    perception.obstacle_map.merge(perception.actors_map)
    return perception, sensor


def _scalar_compute_fov_polygon(perception):
    """Reference implementation from immediately before fan vectorization."""
    transform = perception.sensor.get_transform()
    origin = shapely.Point(transform.location.x, transform.location.y)
    shapely.prepare(origin)

    rays_geometry = shapely.affinity.rotate(
        perception.rays_mls,
        transform.rotation.yaw,
        origin=(0, 0),
        use_radians=False,
    )
    rays_geometry = shapely.affinity.translate(
        rays_geometry,
        xoff=transform.location.x,
        yoff=transform.location.y,
        zoff=0.0,
    )
    rays = shapely.get_parts(rays_geometry)
    tree = shapely.STRtree(perception.obstacle_map.get_all_shapes())

    polygon_points = []
    ideal_points = []
    hit_ray_indices, hit_geometry_indices = tree.query(
        rays, predicate="intersects"
    )
    slice_starts = np.searchsorted(
        hit_ray_indices, np.arange(len(rays)), side="left"
    )
    slice_ends = np.searchsorted(
        hit_ray_indices, np.arange(len(rays)), side="right"
    )

    for ray_index, ray in enumerate(rays):
        indices = hit_geometry_indices[
            slice_starts[ray_index] : slice_ends[ray_index]
        ]
        point = ray.coords[1]
        if len(indices) == 0:
            polygon_points.append(point)
            ideal_points.append(point)
            continue
        if len(indices) == 1:
            geometry = tree.geometries[indices[0]]
        else:
            candidates = shapely.STRtree(tree.geometries.take(indices))
            geometry = candidates.geometries.take(candidates.nearest(origin))

        perception.obstacle_map.set_hit(geometry, True)
        perception.actors_map.set_hit(geometry, True)
        intersection = shapely.intersection(ray, geometry)
        type_id = shapely.get_type_id(intersection)
        if type_id == 1:
            point = shapely.Point(intersection.coords[0])
        elif type_id == 0:
            point = intersection

        polygon_points.append(point)
        ideal_points.append(ray.coords[1])

    if perception.fov < perception.MAX_SENSOR_FOV:
        polygon_points.append(origin)
        ideal_points.append(origin)
    else:
        polygon_points.append(polygon_points[0])
        ideal_points.append(ideal_points[0])

    return polygon_points, ideal_points


def _xy(points):
    return tuple(
        (point.x, point.y) if isinstance(point, shapely.Point) else tuple(point)
        for point in points
    )


def _hit_actor_ids(perception):
    return sorted(
        item.obstacle.id
        for item in perception.obstacle_map.map.values()
        if item.hit
    )


def _assert_matches_scalar(geometries, transform, **kwargs):
    scalar, scalar_sensor = _make_perception(geometries, **kwargs)
    vector, vector_sensor = _make_perception(geometries, **kwargs)
    scalar_sensor.transform = transform
    vector_sensor.transform = transform

    scalar_points, scalar_ideal = _scalar_compute_fov_polygon(scalar)
    vector_points, vector_ideal = vector.compute_fov_polygon()

    assert _xy(vector_points) == _xy(scalar_points)
    assert _xy(vector_ideal) == _xy(scalar_ideal)
    assert _hit_actor_ids(vector) == _hit_actor_ids(scalar)
    return vector


def _rectangle(center_x, center_y, extent_x, extent_y, yaw):
    geometry = shapely.box(
        center_x - extent_x,
        center_y - extent_y,
        center_x + extent_x,
        center_y + extent_y,
    )
    return shapely.affinity.rotate(geometry, yaw, origin=(center_x, center_y))


@pytest.mark.parametrize("seed", range(12))
def test_vectorized_fan_matches_scalar_reference(seed):
    rng = np.random.default_rng(seed)
    geometries = [
        _rectangle(
            rng.uniform(-65, 65),
            rng.uniform(-65, 65),
            rng.uniform(0.1, 7),
            rng.uniform(0.1, 7),
            rng.uniform(-180, 180),
        )
        for _ in range(60)
    ]

    _assert_matches_scalar(
        geometries,
        _transform(
            rng.uniform(-10, 10),
            rng.uniform(-10, 10),
            rng.uniform(-180, 180),
        ),
        fov=(360, 270, 90)[seed % 3],
        ray_step=(1, 5, 7, 10)[seed % 4],
        sensor_range=(20, 50, 70)[seed % 3],
    )


def test_identical_ties_match_scalar_without_fallback():
    geometry = _rectangle(10, 0, 1, 2, 0)

    vector = _assert_matches_scalar(
        [geometry] * 8, _transform(0, 0, 0), ray_step=5
    )

    assert vector.tie_groups > 0
    assert vector.tie_fallbacks == 0


def test_nonidentical_ties_use_scalar_fallback():
    geometries = [
        shapely.Polygon([(10, 0), (11, 1), (11, -1)]),
        shapely.Polygon([(10, 0), (12, 2), (12, 1)]),
    ]

    vector = _assert_matches_scalar(
        geometries, _transform(0, 0, 0), ray_step=5
    )

    assert vector.tie_groups > 0
    assert vector.tie_fallbacks > 0
