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
from __future__ import annotations

from typing import Optional


class MeasuredValue:
    def __init__(self, value, uncertainty=0):
        self.value = value
        self.uncertainty = uncertainty


class MeasuredPoint:
    def __init__(self, x, y, z=0, uncertainty=0):
        self.x = x
        self.y = y
        self.z = z
        self.uncertainty = uncertainty


class Measured2dBox:
    def __init__(self, p1, p2, p3, p4, uncertainty=0):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.uncertainty = uncertainty


class Measurement:
    def __init__(self,
                 target_id: int,
                 position: MeasuredPoint,
                 front: MeasuredPoint,
                 v: MeasuredValue,
                 length: Optional[MeasuredValue],
                 width: Optional[MeasuredValue],
                 box2d: Optional[Measured2dBox],
                 semantic_tag: Optional[MeasuredValue] = None):
        """

        :param target_id: unique id for the target, normally we use the simulation global carla.Actor.id
        :param position:
        :param front:
        :param v:
        :param length:
        :param width:
        :param box2d:
        :param semantic_tag:
        """
        self.target_id = target_id
        self.position = position
        self.front = front
        self.v = v
        self.length = length
        self.width = width
        self.box2d = box2d
        self.semantic_tag = semantic_tag
