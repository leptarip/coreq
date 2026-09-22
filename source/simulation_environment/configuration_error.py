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
"""Configuration-validation errors shared across simulator front ends."""


class SimulationConfigurationError(ValueError):
    """Raised when a simulator configuration is rejected before execution."""


def require_valid_configuration(valid: bool, description: str, *, scope: str) -> None:
    """Convert the validators' legacy ``(bool, reason)`` result into an error."""
    if valid:
        return
    reason = str(description).strip() or "validator returned no reason"
    raise SimulationConfigurationError(f"Invalid {scope} configuration: {reason}")
