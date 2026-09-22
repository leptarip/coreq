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
"""
Lightweight EpisodeResult container shared across the approach_one pipeline.

Kept in its own module to avoid importing heavy simulator dependencies (e.g.,
CARLA) when only the result shape is needed.
"""


class EpisodeResult:
    def __init__(self):
        # Consumers expect these keys to exist; initialize empty dictionaries.
        self.result = {"kpi": {}, "safety_m": {}, "extra": {}, "rare_event": {}}
