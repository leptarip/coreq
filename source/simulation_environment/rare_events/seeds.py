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
"""Deterministic, collision-free episode seeds for importance sampling."""

from typing import Iterable, List, Optional

from source.simulation_environment.configuration_error import SimulationConfigurationError


SAMPLER_SEED_ALGORITHM = "lossless-utf8-int-v1"
SAMPLER_SEED_RULE = (
    "int.from_bytes(utf8('simreq-sampler-v1:<configured-seed-or-none>:"
    "<simulation-seed>'), byteorder='big')"
)
_SAMPLER_SEED_DOMAIN = "simreq-sampler-v1"


def _encode_sampler_seed(simulation_seed: int, configured_seed: Optional[int]) -> int:
    """Encode both integer inputs losslessly for Python's arbitrary-size RNG seed."""
    configured_token = "none" if configured_seed is None else str(int(configured_seed))
    payload = (
        f"{_SAMPLER_SEED_DOMAIN}:{configured_token}:{int(simulation_seed)}"
    ).encode("utf-8")
    return int.from_bytes(payload, byteorder="big", signed=False)


def derive_episode_sampler_seeds(
    simulation_seeds: Iterable[int], configured_seed: Optional[int] = None,
) -> List[int]:
    """Return a unique, domain-separated sampler seed for every episode seed."""
    episode_seeds = [int(seed) for seed in simulation_seeds]
    if len(set(episode_seeds)) != len(episode_seeds):
        raise SimulationConfigurationError(
            "Sampler seed derivation requires unique simulation seeds."
        )
    sampler_seeds = [
        _encode_sampler_seed(seed, configured_seed) for seed in episode_seeds
    ]
    if len(set(sampler_seeds)) != len(sampler_seeds):
        raise SimulationConfigurationError(
            "Sampler seed collision detected; refusing statistically dependent episodes."
        )
    return sampler_seeds
