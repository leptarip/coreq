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

import hashlib


class FreshSeedPolicy:
    """
    Deterministic seed generator that avoids reusing seeds across scenarios/designs.
    """

    def __init__(self, base_seed: int = 0):
        self.base_seed = int(base_seed)
        self._counter = 0

    def next_seed(self, design_id: int | str, scenario_name: str) -> int:
        self._counter += 1
        payload = f"{self.base_seed}:{design_id}:{scenario_name}:{self._counter}".encode("utf-8")
        return int(hashlib.sha1(payload).hexdigest()[:8], 16)
