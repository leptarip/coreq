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
import source.simulation_environment.matrix_helper as mh
import math


class MockAttribute:
    def __init__(self, content):
        self.content = content

    def as_int(self):
        if isinstance(self.content, int):
            return self.content

    def as_float(self):
        if isinstance(self.content, float):
            return self.content
        else:
            return float(self.content)


class MockCameraBp:
    def __init__(self, image_size_x, image_size_y, fov=90):
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.fov = fov

    def get_attribute(self, name):
        if name == "image_size_x":
            return MockAttribute(self.image_size_x)
        elif name == "image_size_y":
            return MockAttribute(self.image_size_y)
        elif name == "fov":
            return MockAttribute(self.fov)


class MockVector3D:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def dot(self, other) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z


class MockLocation:
    def __init__(self, pos_x=0, pos_y=0, pos_z=0):
        self.x = pos_x
        self.y = pos_y
        self.z = pos_z


class MockRotation:
    def __init__(self, yaw=0, pitch=0, roll=0):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll


class MockTransform:
    def __init__(self, location=MockLocation(), rotation=MockRotation()):
        self.location = location
        self.rotation = rotation

    def get_transform_matrix(self):
        return mh.get_matrix_from_carla_transform(self)

    def get_forward_vector(self):
        """
        const float cp = std::cos(ToRadians(rotation.pitch));
        const float sp = std::sin(ToRadians(rotation.pitch));
        const float cy = std::cos(ToRadians(rotation.yaw));
        const float sy = std::sin(ToRadians(rotation.yaw));
        return {cy * cp, sy * cp, sp};
        :return:
        """
        forward = MockVector3D()
        cp = math.cos(math.radians(self.rotation.pitch))
        sp = math.sin(math.radians(self.rotation.pitch))
        cy = math.cos(math.radians(self.rotation.yaw))
        sy = math.sin(math.radians(self.rotation.yaw))

        forward.x = cp * cy
        forward.y = cp * sy
        forward.z = sp

        return forward


class MockWayPoint:
    def __init__(self, transform=MockTransform()):
        self.transform = transform


class MockExtent:
    def __init__(self, extent_x=0, extent_y=0, extent_z=0):
        self.x = extent_x
        self.y = extent_y
        self.z = extent_z


class MockBoundingBox:
    def __init__(self, location, extent_x=0, extent_y=0, extent_z=0):
        self.location = location
        self.extent = MockExtent(extent_x, extent_y, extent_z)


class MockSensor:
    def __init__(self, pos_x=0, pos_y=0, pos_z=10, yaw=0, pitch=0, roll=0):
        self.location = MockLocation(pos_x, pos_y, pos_z)
        self.rotation = MockRotation(yaw, pitch, roll)
        self.transform = MockTransform(self.location, self.rotation)

    def get_transform(self):
        return self.transform

    def get_location(self):
        return self.location


class MockBuilding:
    def __init__(self, pos_x=0, pos_y=0, extent_x=0, extent_y=0, extent_z=0):
        self.location = MockLocation(pos_x, pos_y)
        self.bounding_box = MockBoundingBox(self.location, extent_x, extent_y, extent_z)
