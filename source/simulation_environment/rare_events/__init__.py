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
Scaffolding for rare-event estimation (importance sampling helpers).

These utilities are intentionally self-contained and not wired into the main
simulation yet; they exist so you can import them when you are ready to
integrate importance sampling into the simulator.
"""

from .wrapper import ImportanceSampler

__all__ = [
    "ImportanceSampler",
]
