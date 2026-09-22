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

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .episode_manager import EpisodeManager


def _row_signature(row: Dict[str, object]) -> str:
    return str(row["ep_uid"])


class MergedEpisodeManager:
    """
    Read cached episodes from multiple EpisodeManager roots while writing only to
    the primary manager for the current run.
    """

    def __init__(self, write_manager: EpisodeManager, managers: Sequence[EpisodeManager]):
        self.write_manager = write_manager
        self.managers = list(managers)
        self.GRID = write_manager.GRID
        self.keys = write_manager.keys
        self.space_spec = write_manager.space_spec
        self.scenario = write_manager.scenario
        self.root_dir = write_manager.root_dir
        self.dir_sc = write_manager.dir_sc
        self._consumed_signatures: set[str] = set()
        self._counts_cache: Optional[Dict[int, int]] = None

    def episodes(self, gidx: int, consume: bool = False, limit: Optional[int] = None) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        seen_signatures: set[str] = set()
        for manager in self.managers:
            for row in manager.episodes(int(gidx), consume=False):
                sig = _row_signature(row)
                if sig in seen_signatures:
                    continue
                if consume and sig in self._consumed_signatures:
                    continue
                seen_signatures.add(sig)
                rows.append(row)
                if consume:
                    self._consumed_signatures.add(sig)
                if limit is not None and len(rows) >= int(limit):
                    return rows
        return rows

    def list_gidx_with_counts(self) -> Dict[int, int]:
        if self._counts_cache is not None:
            return dict(self._counts_cache)
        keys: set[int] = set()
        for manager in self.managers:
            keys.update(int(gi) for gi in manager.list_gidx_with_counts().keys())
        self._counts_cache = {gi: len(self.episodes(gi, consume=False)) for gi in sorted(keys)}
        return dict(self._counts_cache)

    def count(self, gidx: int, unconsumed_only: bool = False) -> int:
        if not unconsumed_only:
            counts = self.list_gidx_with_counts()
            return int(counts.get(int(gidx), 0))
        rows = self.episodes(int(gidx), consume=False)
        return sum(1 for row in rows if _row_signature(row) not in self._consumed_signatures)

    def append_batch(self, gidx: int, outputs: List[object], meta) -> List[str]:
        self._counts_cache = None
        return self.write_manager.append_batch(gidx, outputs, meta)

    def append_one(self, gidx: int, payload: object, meta) -> str:
        self._counts_cache = None
        return self.write_manager.append_one(gidx, payload, meta)

    def flush(self) -> None:
        self._counts_cache = None
        self.write_manager.flush()

    def summarize_shards(self):
        return self.write_manager.summarize_shards()


def build_merged_episode_manager(
    *,
    write_root_dir: str,
    scenario_name: str,
    space_spec,
    GRID: np.ndarray,
    flush_every: int,
    episode_cache_roots: Optional[Sequence[str]] = None,
) -> MergedEpisodeManager:
    write_manager = EpisodeManager(
        root_dir=write_root_dir,
        scenario_name=scenario_name,
        space_spec=space_spec,
        GRID=GRID,
        flush_every=flush_every,
    )
    managers: List[EpisodeManager] = []
    seen: set[str] = set()
    write_root_abs = os.path.abspath(str(write_manager.dir_sc))
    for root in [*(episode_cache_roots or []), write_root_abs]:
        if not root:
            continue
        root_abs = os.path.abspath(str(root))
        if root_abs in seen:
            continue
        seen.add(root_abs)
        if root_abs == write_root_abs:
            managers.append(write_manager)
            continue
        managers.append(
            EpisodeManager(
                root_dir=root_abs,
                scenario_name=scenario_name,
                space_spec=space_spec,
                GRID=GRID,
                flush_every=flush_every,
            )
        )
    merged = MergedEpisodeManager(write_manager, managers)
    merged.source_dirs = [getattr(manager, "dir_sc", getattr(manager, "root_dir", "")) for manager in managers]
    return merged
