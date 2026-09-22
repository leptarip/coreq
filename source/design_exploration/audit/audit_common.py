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

from source.design_exploration.commons.offline_labeling import _payload_to_episode_result
from source.design_exploration.commons.episodes.episode_result import EpisodeResult


def pool_record_to_episode_result(rec, *, require_min_d: bool = True) -> EpisodeResult:
    """Decode a completion record, optionally leaving metric policy to the caller."""
    if rec.status != "done":
        raise RuntimeError(f"Job {rec.job_id} failed after {rec.attempts} attempts: {rec.error}")
    payload = rec.payload
    ep_raw = payload[0] if isinstance(payload, list) and payload else payload
    if not ep_raw:
        raise RuntimeError(f"Job {rec.job_id} returned empty payload")
    ep_res = ep_raw if isinstance(ep_raw, EpisodeResult) else _payload_to_episode_result(
        ep_raw if isinstance(ep_raw, dict) else ep_raw
    )
    safety = getattr(ep_res, "result", {}).get("safety_m", {}) or {}
    if require_min_d and ("min_d" not in safety or safety.get("min_d") is None):
        raise RuntimeError(f"Job {rec.job_id} payload missing min_d: {safety}")
    return ep_res

