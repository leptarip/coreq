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

from typing import Any, Dict, List, Optional, Protocol

from source.design_exploration.commons.episodes.episode_result import EpisodeResult


class SimulationInterface(Protocol):
    """
    Structural interface for simulator adapters used by active-learning code.
    """

    output_folder: str
    algo_version: str
    num_carla_instances: int
    carla_path: Optional[str]

    def simulate(
        self,
        scenario_name: str,
        base_cfg: Dict[str, Any],
        design: Any,
        num_episodes: int,
    ) -> List[EpisodeResult]:
        ...

    def close(self) -> None:
        ...
