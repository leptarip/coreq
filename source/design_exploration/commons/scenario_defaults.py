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

from pathlib import Path
from typing import Dict

_HERE = Path(__file__).resolve()
_REPO_ROOT = next((cand for cand in [_HERE] + list(_HERE.parents) if (cand / "source").is_dir()), _HERE.parent)
_CONFIG_ROOT = _REPO_ROOT / "configuration"

SCENARIO_1: Dict[str, str] = {
    "name": "intersection1",
    "config": str(_CONFIG_ROOT / "config_intersection1_debug.toml"),
}

SCENARIO_2: Dict[str, str] = {
    "name": "intersection2",
    "config": str(_CONFIG_ROOT / "config_intersection2_debug.toml"),
}

