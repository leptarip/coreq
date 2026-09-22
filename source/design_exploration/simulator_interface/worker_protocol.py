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
"""Failure and seed contracts shared by CARLA workers and their orchestrator."""

from typing import Any, List, NoReturn, Optional, Sequence

from source.simulation_environment.configuration_error import SimulationConfigurationError


CONFIGURATION_ERROR_REASON = "configuration_error"


def requested_episode_seeds(
    *,
    seed: Optional[int],
    seeds: Optional[Sequence[int]],
    num_episodes: int,
) -> Optional[List[int]]:
    """Return the exact per-episode seed plan represented by a pool request."""
    if seed is not None and seeds is not None:
        raise SimulationConfigurationError(
            "A worker job must provide either seed or seeds, not both"
        )
    episodes = int(num_episodes)
    if episodes < 1:
        raise SimulationConfigurationError(
            f"A worker job must request at least one episode, got {episodes}"
        )
    if seeds is not None:
        seed_list = [int(value) for value in seeds]
        if len(seed_list) != episodes:
            raise SimulationConfigurationError(
                "Explicit seed count must match num_episodes "
                f"(got {len(seed_list)} and {episodes})"
            )
        return seed_list
    if seed is None:
        return None
    base_seed = int(seed)
    return [base_seed + index for index in range(episodes)]


def _result_dict(payload: Any):
    if hasattr(payload, "result"):
        result = getattr(payload, "result", {}) or {}
        return result if isinstance(result, dict) else None
    if isinstance(payload, dict):
        nested = payload.get("result")
        if isinstance(nested, dict):
            return nested
        return payload
    return None


def verified_simulator_seed(payload: Any, requested_seed: int) -> int:
    """Require runner evidence that an episode consumed its requested seed."""
    result = _result_dict(payload)
    meta = (result or {}).get("meta", {}) or {}
    simulator_seed = meta.get("simulator_seed") if isinstance(meta, dict) else None
    if simulator_seed is None:
        raise SimulationConfigurationError(
            "Simulator result lacks runner-reported meta.simulator_seed "
            f"for requested seed {int(requested_seed)}"
        )
    try:
        actual_seed = int(simulator_seed)
    except (TypeError, ValueError) as exc:
        raise SimulationConfigurationError(
            f"Simulator reported a non-integer seed: {simulator_seed!r}"
        ) from exc
    if actual_seed != int(requested_seed):
        raise SimulationConfigurationError(
            "Simulator consumed seed "
            f"{actual_seed}, expected requested seed {int(requested_seed)}"
        )
    return actual_seed


def attach_verified_seed_metadata(payload: Any, requested_seed: int) -> Any:
    """Validate runner seed evidence and add unambiguous request metadata."""
    actual_seed = verified_simulator_seed(payload, requested_seed)
    if hasattr(payload, "result"):
        result_copy = dict(getattr(payload, "result", {}) or {})
        meta = dict(result_copy.get("meta", {}) or {})
        meta["seed"] = actual_seed
        meta["requested_seed"] = int(requested_seed)
        result_copy["meta"] = meta
        payload.result = result_copy
        return payload
    if isinstance(payload, dict):
        payload_copy = dict(payload)
        nested = payload_copy.get("result")
        if isinstance(nested, dict):
            nested_copy = dict(nested)
            meta = dict(nested_copy.get("meta", {}) or {})
            meta["seed"] = actual_seed
            meta["requested_seed"] = int(requested_seed)
            nested_copy["meta"] = meta
            payload_copy["result"] = nested_copy
        else:
            meta = dict(payload_copy.get("meta", {}) or {})
            meta["seed"] = actual_seed
            meta["requested_seed"] = int(requested_seed)
            payload_copy["meta"] = meta
        return payload_copy
    raise SimulationConfigurationError(
        f"Unsupported simulator payload type for seed verification: {type(payload).__name__}"
    )


def attach_verified_seed_plan(
    payload: Any,
    *,
    seed: Optional[int],
    seeds: Optional[Sequence[int]],
    num_episodes: int,
) -> Any:
    """Validate every seeded episode while preserving the payload container type."""
    expected = requested_episode_seeds(
        seed=seed,
        seeds=seeds,
        num_episodes=num_episodes,
    )
    if expected is None:
        return payload
    payload_is_list = isinstance(payload, list)
    episodes = payload if payload_is_list else [payload]
    if len(episodes) != len(expected):
        raise SimulationConfigurationError(
            "Simulator result count does not match requested seed plan "
            f"(got {len(episodes)} and {len(expected)})"
        )
    verified = [
        attach_verified_seed_metadata(episode, requested_seed)
        for episode, requested_seed in zip(episodes, expected)
    ]
    return verified if payload_is_list else verified[0]


def is_retryable_failure(reason: Optional[str]) -> bool:
    """Configuration defects are deterministic; existing failures may retry."""
    return reason != CONFIGURATION_ERROR_REASON


def should_retry_failure(reason: Optional[str], *, attempts: int, max_retries: int) -> bool:
    """Apply the pool's retry budget only to retryable failure classes."""
    return is_retryable_failure(reason) and int(attempts) <= int(max_retries)


def failure_reason_for_exception(exc: Exception) -> str:
    """Classify deterministic configuration failures at the worker boundary."""
    if isinstance(exc, SimulationConfigurationError):
        return CONFIGURATION_ERROR_REASON
    return "exception"


def raise_worker_failure(*, reason: Optional[str], error: Optional[str], job_id: str) -> NoReturn:
    """Restore typed worker failures at the public simulator interface."""
    if reason == CONFIGURATION_ERROR_REASON:
        raise SimulationConfigurationError(error or "Simulator configuration rejected")
    raise RuntimeError(f"Worker job failed (job_id={job_id}, error={error})")
