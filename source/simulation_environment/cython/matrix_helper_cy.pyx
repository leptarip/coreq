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
import math

import numpy as np
import carla


def get_matrix_from_carla_transform(carla_transform : carla.Transform):
    """
    Creates matrix from carla transform.
    """
    rotation = carla_transform.rotation
    location = carla_transform.location
    cdef float c_y = np.cos(np.radians(rotation.yaw))
    cdef float s_y = np.sin(np.radians(rotation.yaw))
    cdef float c_r = np.cos(np.radians(rotation.roll))
    cdef float s_r = np.sin(np.radians(rotation.roll))
    cdef float c_p = np.cos(np.radians(rotation.pitch))
    cdef float s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.identity(4)
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def get_matrix_from_carla_location(carla_location: carla.Location):
    """
    Creates matrix from carla transform.
    """
    location = carla_location
    cdef float c_y = np.cos(np.radians(0))
    cdef float s_y = np.sin(np.radians(0))
    cdef float c_r = np.cos(np.radians(0))
    cdef float s_r = np.sin(np.radians(0))
    cdef float c_p = np.cos(np.radians(0))
    cdef float s_p = np.sin(np.radians(0))
    matrix = np.identity(4)
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def get_matrix(yaw: float, roll: float, pitch: float, x: float, y: float, z: float):
    """
    Creates matrix from carla transform.
    """
    cdef float c_y = np.cos(np.radians(yaw))
    cdef float s_y = np.sin(np.radians(yaw))
    cdef float c_r = np.cos(np.radians(roll))
    cdef float s_r = np.sin(np.radians(roll))
    cdef float c_p = np.cos(np.radians(pitch))
    cdef float s_p = np.sin(np.radians(pitch))
    matrix = np.identity(4)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def rotate_reference(reference, axe: str, angle_deg: float):
    if axe == "x":
        return np.dot(reference, get_rotation_x(angle_deg))
    elif axe == "y":
        return np.dot(reference, get_rotation_y(angle_deg))
    elif axe == "z":
        return np.dot(reference, get_rotation_z(angle_deg))

    return None


def translate_reference(reference, x=0, y=0, z=0):
    matrix = np.identity(4)

    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    return np.dot(reference, matrix)


def get_rotation_x(roll: float):
    matrix = np.zeros((4, 4))

    cdef float a = np.radians(roll)
    cdef float cos_a = math.cos(a)
    cdef float sin_a = math.sin(a)

    matrix[3, 3] = 1  # always 1

    matrix[0, 0] = 1
    matrix[1, 1] = cos_a
    matrix[1, 2] = -sin_a

    matrix[2, 1] = sin_a
    matrix[2, 2] = cos_a

    return matrix


def get_rotation_y(pitch: float):
    matrix = np.zeros((4, 4))

    cdef float a = np.radians(pitch)
    cdef float cos_a = math.cos(a)
    cdef float sin_a = math.sin(a)

    matrix[3, 3] = 1  # always 1

    matrix[0, 0] = cos_a
    matrix[0, 2] = sin_a

    matrix[1, 1] = 1

    matrix[2, 0] = -sin_a
    matrix[2, 2] = cos_a

    return matrix


def get_rotation_z(yaw: float):
    matrix = np.zeros((4, 4))

    cdef float a = np.radians(yaw)
    cdef float cos_a = math.cos(a)
    cdef float sin_a = math.sin(a)

    matrix[3, 3] = 1  # always 1

    matrix[0, 0] = cos_a
    matrix[0, 1] = -sin_a

    matrix[1, 0] = sin_a
    matrix[1, 1] = cos_a

    matrix[2, 2] = 1

    return matrix
