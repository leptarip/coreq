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
# episode_manager.py
from __future__ import annotations
import os, re, time, json, glob, uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Any, Tuple
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from source.design_exploration.commons.design_space import SpaceSpec, Design

EP_UID_COLUMN = "ep_uid"


def new_episode_uid() -> str:
    """
    Durable identifier for a newly simulated episode.

    Random rather than sequential: ``ep_id`` is a per-manager counter, so two
    managers opened on one store issue the same ids. A random uid needs no
    coordination between processes, workers or threads.
    """
    return uuid.uuid4().hex

# ----------------------- utilities -----------------------

def _round_tuple(vals: Iterable[float], ndigits: int = 8) -> Tuple[float, ...]:
    return tuple(round(float(x), ndigits) for x in vals)

def _design_to_tuple(design: Design, keys: Tuple[str, ...]) -> Tuple[float, ...]:
    return tuple(float(design[k]) for k in keys)

def _episode_to_payload_dict(ep_out: Any) -> Dict[str, Any]:
    """
    Convert a simulator episode output to a JSON-serializable dict.
    We do NOT evaluate/pass/fail here: pure caching only.
    """
    if isinstance(ep_out, dict):
        return ep_out
    if hasattr(ep_out, "result"):  # common pattern in your sim wrapper
        return ep_out.result if isinstance(ep_out.result, dict) else {"result": str(ep_out.result)}

def _safe_json_dump(obj: Any) -> str:
    try:
        return json.dumps(obj, allow_nan=True)
    except Exception:
        # fallback: stringify non-serializable bits
        def _default(o):
            try:
                return dict(o)
            except Exception:
                return str(o)
        return json.dumps(obj, default=_default, allow_nan=True)

def _safe_json_load(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return {"payload_raw": s}


def _is_missing_scalar(value: Any) -> bool:
    """Return whether a scalar parquet value is missing."""
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    return bool(missing) if np.isscalar(missing) else False


def _episode_identity(row: pd.Series) -> Tuple[Any, ...]:
    """Return the durable key used when compacting one episode row."""
    uid = row.get(EP_UID_COLUMN)
    if _is_missing_scalar(uid):
        raise RuntimeError(f"Cannot compact a row without '{EP_UID_COLUMN}'.")
    return ("uid", str(uid))


def _deduplicate_for_compaction(
    frame: pd.DataFrame, source_names: List[str]
) -> Tuple[pd.DataFrame, int]:
    """
    Remove only byte-for-byte-equivalent repetitions of one episode identity.

    Rows are identified by ``ep_uid``; equal ``ep_id`` values are not
    duplicates. If one identity names conflicting content, the compaction aborts
    instead of choosing a row and destroying evidence.
    """
    if len(frame) != len(source_names):
        raise ValueError("source_names must identify every row in frame")

    first_position: Dict[Tuple[Any, ...], int] = {}
    keep_positions: List[int] = []
    duplicates = 0
    for position in range(len(frame)):
        row = frame.iloc[position]
        key = _episode_identity(row)
        first = first_position.get(key)
        if first is None:
            first_position[key] = position
            keep_positions.append(position)
            continue
        if not frame.iloc[first].equals(row):
            raise RuntimeError(
                "Conflicting rows share episode identity "
                f"{key!r}: {source_names[first]!r} and {source_names[position]!r}. "
                "No source shards were deleted."
            )
        duplicates += 1

    return frame.iloc[keep_positions].reset_index(drop=True), duplicates

# ----------------------- metadata -----------------------

@dataclass(frozen=True)
class EpisodeMeta:
    scenario: str
    split: str      # e.g., "train", "AUDIT", or "VALIDATION"
    provenance: str # free text describing the workflow/source of the episode.

# ----------------------- manager -----------------------

class EpisodeManager:
    """
    Episode cache / logger keyed by grid index (gidx).

    Goals:
    1) Resume after interruptions without re-simulating already-run episodes.
    2) Reuse episodes across multiple audit thresholds (no double sim).
    3) Keep a clean separation: this class NEVER evaluates pass/fail.

    Key features:
    - Progressive global episode ids (ep_id).
    - Sharded parquet files with progressive file index.
    - On init: load ALL shards under root for this scenario.
    - Per-gidx FIFO queues; retrieve with consume={False,True} to control i.i.d. usage.

    Parquet schema per row:
        ep_uid:str, ep_id:int, sc_name:str, gidx:int, stamp:int(ns),
        <param columns...>,
        split:str, provenance:str,
        payload_json:str
    """

    def __init__(self,
                 root_dir: str,
                 scenario_name: str,
                 space_spec: SpaceSpec,
                 GRID: np.ndarray,
                 shard_prefix: str = "episodes",
                 summary_prefix: str = "summary",
                 flush_every: int = 200,
                 on_duplicate: str = "warn"):
        # layout: <root_dir>/<scenario_name>/*.parquet
        self.root_dir = os.path.abspath(root_dir)
        self.scenario = scenario_name
        # Allow caller to supply an already scenario-scoped episodes directory to avoid
        # double-appending the scenario name.
        base = os.path.basename(self.root_dir)
        parent_base = os.path.basename(os.path.dirname(self.root_dir))
        if (base == "episodes" and parent_base == self.scenario) or base == self.scenario:
            self.dir_sc = self.root_dir
        else:
            self.dir_sc = os.path.join(self.root_dir, self.scenario)
        os.makedirs(self.dir_sc, exist_ok=True)

        self.space_spec = space_spec
        self.keys: Tuple[str, ...] = tuple(space_spec.keys)
        self.GRID = np.asarray(GRID, dtype=float)

        # map rounded param tuple -> gidx
        self._param_to_gidx: Dict[Tuple[float, ...], int] = {
            _round_tuple(row): int(i) for i, row in enumerate(self.GRID)
        }

        if on_duplicate not in ("warn", "raise"):
            raise ValueError(f"on_duplicate must be 'warn' or 'raise', got {on_duplicate!r}")
        self.on_duplicate = on_duplicate
        self.flush_every = int(flush_every)
        self.shard_prefix = shard_prefix
        self.summary_prefix = summary_prefix

        # in-memory store
        self._queues: Dict[int, deque] = defaultdict(deque)  # gidx -> deque of dict rows (FIFO)
        self._consumed_epids: set[int] = set()               # track consumed (when consume=True)
        self._buf: List[Dict[str, Any]] = []                 # write buffer
        self._since_flush = 0

        # progressive counters
        self._next_ep_id: int = 1
        self._next_file_idx: int = 1

        self._load_all_existing()

    # ------------------- Public API -------------------

    def count(self, gidx: int, unconsumed_only: bool = False) -> int:
        q = self._queues.get(int(gidx), deque())
        if not unconsumed_only:
            return len(q)
        return sum(1 for r in q if r["ep_id"] not in self._consumed_epids)

    def list_gidx_with_counts(self) -> Dict[int, int]:
        return {gi: len(q) for gi, q in self._queues.items()}

    def episodes(self, gidx: int, consume: bool = False, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get cached episodes for a design index in FIFO order.
        - consume=False: returns a copy; does NOT mark as consumed (good for multi-threshold reuse).
        - consume=True: marks returned ep_ids as consumed (good for resume without double-counting).
        """
        gi = int(gidx)
        q = self._queues.get(gi, deque())
        out: List[Dict[str, Any]] = []
        n = len(q) if limit is None else min(limit, len(q))
        taken = 0
        for row in q:
            if consume and row["ep_id"] in self._consumed_epids:
                continue
            out.append(_materialize_row(row))
            if consume:
                self._consumed_epids.add(row["ep_id"])
            taken += 1
            if taken >= n:
                break
        return out

    def first(self, gidx: int, consume: bool = False) -> Optional[Dict[str, Any]]:
        eps = self.episodes(gidx, consume=consume, limit=1)
        return eps[0] if eps else None

    def append_one(self, gidx: int, payload: Any, meta: EpisodeMeta) -> str:
        """
        Append a SINGLE new episode result and return its durable episode UID.
        """
        return self.append_batch(gidx, [payload], meta)[0]

    def append_batch(self, gidx: int, outputs: List[Any], meta: EpisodeMeta) -> List[str]:
        """
        Append a batch of new episodes for a given gidx. Assigns progressive ep_id.
        Stores full payload as JSON (no evaluation).
        """
        gi = int(gidx)
        stamp = time.time_ns()
        params = {k: float(v) for k, v in zip(self.keys, self.GRID[gi])}
        episode_uids: List[str] = []

        for ep in outputs:
            payload_dict = _episode_to_payload_dict(ep)
            ep_uid = new_episode_uid()
            row = dict(
                ep_uid=ep_uid,
                ep_id=self._next_ep_id,
                sc_name=self.scenario,
                gidx=gi,
                stamp=stamp,
                split=meta.split,
                provenance=meta.provenance,
                payload_json=_safe_json_dump(payload_dict),
                **params,
            )
            self._buf.append(row)
            # store a lightweight in-memory copy; payload parsed lazily at retrieval
            row_mem = row.copy()
            row_mem["label_mode"] = meta.provenance
            self._queues[gi].append(row_mem)
            self._next_ep_id += 1
            self._since_flush += 1
            episode_uids.append(ep_uid)

        if self._since_flush >= self.flush_every:
            self.flush()
        return episode_uids

    def flush(self):
        if not self._buf:
            return
        df = pd.DataFrame(self._buf)
        # write to a NEW shard file with progressive index
        shard = os.path.join(self.dir_sc, f"{self.shard_prefix}.{self._next_file_idx:06d}.parquet")
        tmp = shard + ".tmp"
        df.to_parquet(tmp, engine="auto", compression="snappy", index=False)
        os.replace(tmp, shard)
        self._buf.clear()
        self._since_flush = 0
        self._next_file_idx += 1

    def summarize_shards(self) -> Optional[str]:
        """
        Losslessly compact episode shards into one verified summary parquet.

        Episode identity is ``ep_uid``. Equal ``ep_id`` values alone are preserved. A repeated
        identity is removed only when every stored column agrees; conflicting
        content aborts compaction. Every shard must be readable, and the written
        summary is round-trip verified before any source shard is deleted.

        The summary file name is progressive (summary.000001.parquet, ...).
        Returns the summary path or None if no shards were found.
        """
        # ensure any buffered rows are persisted before summarizing
        self.flush()

        shard_pattern = os.path.join(self.dir_sc, f"{self.shard_prefix}.*.parquet")
        shard_paths = sorted(glob.glob(shard_pattern))
        if not shard_paths:
            return None

        dfs: List[pd.DataFrame] = []
        source_names: List[str] = []
        for p in shard_paths:
            try:
                shard = pd.read_parquet(p)
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot read episode shard {p!r}; compaction aborted and no "
                    "source shards were deleted."
                ) from exc
            dfs.append(shard)
            source_names.extend([os.path.basename(p)] * len(shard))

        if not dfs:
            return None

        df = pd.concat(dfs, ignore_index=True)
        if "sc_name" in df.columns:
            scenarios = {str(v) for v in df["sc_name"].dropna().unique()}
            if scenarios and scenarios != {self.scenario}:
                raise RuntimeError(
                    f"Episode shards under {self.dir_sc!r} contain scenarios "
                    f"{sorted(scenarios)!r}; expected only {self.scenario!r}. "
                    "No source shards were deleted."
                )
        df, duplicate_count = _deduplicate_for_compaction(df, source_names)
        print(
            "[EpisodeManager] summarize_shards: found "
            f"{duplicate_count} repeated episode identities."
        )

        next_summary_idx = self._next_summary_idx()
        summary_path = os.path.join(self.dir_sc, f"{self.summary_prefix}.{next_summary_idx:06d}.parquet")
        tmp = summary_path + ".tmp"
        try:
            df.to_parquet(tmp, engine="auto", compression="snappy", index=False)
            written = pd.read_parquet(tmp)
            pd.testing.assert_frame_equal(
                df.reset_index(drop=True),
                written.reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
            )
            os.replace(tmp, summary_path)
        except Exception as exc:
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
            raise RuntimeError(
                f"Could not write and verify summary {summary_path!r}; no source "
                "shards were deleted."
            ) from exc

        delete_errors: List[Tuple[str, str]] = []
        for p in shard_paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
            except OSError as exc:
                delete_errors.append((p, str(exc)))

        if delete_errors:
            details = "; ".join(f"{path}: {error}" for path, error in delete_errors)
            raise RuntimeError(
                f"Summary {summary_path!r} was written and verified, but some source "
                f"shards could not be deleted: {details}"
            )

        # Do not assume compaction had exclusive access to the directory. A
        # concurrently-created shard remains in place and determines the next
        # available file index rather than being overwritten by index 1.
        remaining = sorted(glob.glob(shard_pattern))
        shard_regex = re.compile(rf"{re.escape(self.shard_prefix)}\.(\d+)\.parquet$")
        remaining_indices = []
        for path in remaining:
            match = shard_regex.search(os.path.basename(path))
            if match:
                remaining_indices.append(int(match.group(1)))
        self._next_file_idx = max(remaining_indices, default=0) + 1
        return summary_path

    # ------------------- Internal: loading -------------------

    def _load_all_existing(self):
        """
        Load every ``episodes.*.parquet`` and ``summary.*.parquet`` shard for
        this scenario into per-gidx FIFO queues ordered by ``ep_id``.

        Every row must carry ``ep_uid``, ``ep_id``, ``gidx``, ``payload_json``
        and the design parameters, and its parameters must equal ``GRID[gidx]``.
        Anything else raises: stores are expected in exactly this format.
        """
        shard_paths = sorted(glob.glob(os.path.join(self.dir_sc, f"{self.shard_prefix}.*.parquet")))
        summary_paths = sorted(glob.glob(os.path.join(self.dir_sc, f"{self.summary_prefix}.*.parquet")))
        required = {EP_UID_COLUMN, "ep_id", "gidx", "payload_json", *self.keys}
        file_idx_regex = re.compile(rf"{re.escape(self.shard_prefix)}\.(\d+)\.parquet$")

        max_ep = 0
        max_file_idx = 0
        rows: List[Dict[str, Any]] = []
        for path in [*shard_paths, *summary_paths]:
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                raise RuntimeError(f"Cannot read episode shard {path!r}.") from exc
            missing = required.difference(df.columns)
            if missing:
                raise RuntimeError(f"Episode shard {path!r} is missing columns {sorted(missing)}.")
            if "sc_name" in df.columns:
                df = df[df["sc_name"] == self.scenario]
            gidx = df["gidx"].to_numpy(dtype=int)
            if gidx.size and (gidx.min() < 0 or gidx.max() >= len(self.GRID)):
                raise RuntimeError(f"Episode shard {path!r} has gidx outside the design grid.")
            params = df[list(self.keys)].to_numpy(dtype=float)
            if not np.allclose(params, self.GRID[gidx], rtol=0.0, atol=1e-8):
                raise RuntimeError(
                    f"Episode shard {path!r} has design parameters that disagree with their gidx."
                )
            for record in df.to_dict("records"):
                if "provenance" not in record and "label_mode" in record:
                    record["provenance"] = record["label_mode"]
                if "label_mode" not in record and "provenance" in record:
                    record["label_mode"] = record["provenance"]
                max_ep = max(max_ep, int(record["ep_id"]))
                rows.append(record)
            match = file_idx_regex.search(os.path.basename(path))
            if match:
                max_file_idx = max(max_file_idx, int(match.group(1)))

        # ep_id is a per-manager counter, so it cannot identify an episode across
        # stores; ep_uid can. Rows repeated under one ep_uid (a shard and the
        # summary it was compacted into) are loaded once.
        rows.sort(key=lambda rr: int(rr["ep_id"]))
        seen: set[str] = set()
        repeated = 0
        for rr in rows:
            uid = str(rr[EP_UID_COLUMN])
            if uid in seen:
                repeated += 1
                continue
            seen.add(uid)
            self._queues[int(rr["gidx"])].append(dict(rr))
        if repeated:
            message = (
                f"[EpisodeManager] {self.scenario}: {repeated} rows repeat an ep_uid "
                "and were loaded once."
            )
            if self.on_duplicate == "raise":
                raise RuntimeError(message)
            print(message)

        self._next_ep_id = max_ep + 1
        self._next_file_idx = max_file_idx + 1

    # ------------------- convenience -------------------

    def _next_summary_idx(self) -> int:
        summary_pattern = os.path.join(self.dir_sc, f"{self.summary_prefix}.*.parquet")
        summary_regex = re.compile(rf"{re.escape(self.summary_prefix)}\.(\d+)\.parquet$")
        max_idx = 0
        for p in glob.glob(summary_pattern):
            m = summary_regex.search(os.path.basename(p))
            if not m:
                continue
            try:
                max_idx = max(max_idx, int(m.group(1)))
            except Exception:
                continue
        return max_idx + 1

def _materialize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a materialized copy with decoded payload.
    (Keeps original columns; adds 'payload' key with the parsed dict.)
    """
    out = dict(row)
    out["payload"] = _safe_json_load(out.pop("payload_json"))
    return out
