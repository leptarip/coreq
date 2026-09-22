# Copyright (c) 2026 278097159+leptarip@users.noreply.github.com
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

"""Regression tests for lossless, transactional episode compaction."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from source.design_exploration.commons.design_space import DiscreteSpace, SpaceSpec
from source.design_exploration.commons.episodes.episode_manager import EpisodeManager, EpisodeMeta


SPEC = SpaceSpec(name="test", spaces={"x": DiscreteSpace((0.0, 1.0))})
GRID = SPEC.build_grid()
SCENARIO = "scenario"


def _row(*, ep_uid=None, ep_id=1, stamp=100, x=0.0, payload="a"):
    row = {
        "ep_id": ep_id,
        "sc_name": SCENARIO,
        "gidx": int(x),
        "stamp": stamp,
        "split": "train",
        "provenance": "RAW",
        "payload_json": '{"value":"%s"}' % payload,
        "x": x,
    }
    if ep_uid is not None:
        row["ep_uid"] = ep_uid
    return row


def _write_shard(root: Path, index: int, rows):
    scenario_dir = root / SCENARIO
    scenario_dir.mkdir(parents=True, exist_ok=True)
    path = scenario_dir / f"episodes.{index:06d}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _manager(root: Path):
    return EpisodeManager(
        root_dir=str(root),
        scenario_name=SCENARIO,
        space_spec=SPEC,
        GRID=GRID,
        on_duplicate="warn",
    )


def test_append_returns_distinct_durable_episode_uids(tmp_path):
    manager = _manager(tmp_path)
    meta = EpisodeMeta(scenario=SCENARIO, split="AUDIT", provenance="test")

    first = manager.append_one(0, {"safety_m": {"min_d": 1.0}}, meta)
    second = manager.append_one(0, {"safety_m": {"min_d": 1.0}}, meta)
    rows = manager.episodes(0, consume=False)

    assert first != second
    assert [row["ep_uid"] for row in rows] == [first, second]
    assert all("cycle" not in row for row in rows)
    manager.flush()
    shard = next((tmp_path / SCENARIO).glob("episodes.*.parquet"))
    stored = pd.read_parquet(shard)
    assert "cycle" not in stored.columns
    assert stored["split"].tolist() == ["AUDIT", "AUDIT"]
    assert stored["provenance"].tolist() == ["test", "test"]
    assert [row["ep_uid"] for row in _manager(tmp_path).episodes(0)] == [first, second]


def test_compaction_preserves_distinct_uids_that_share_ep_id(tmp_path):
    first = _write_shard(
        tmp_path, 1, [_row(ep_uid="uid-a", ep_id=7, stamp=100, payload="a")]
    )
    second = _write_shard(
        tmp_path, 2, [_row(ep_uid="uid-b", ep_id=7, stamp=200, payload="b")]
    )

    summary = Path(_manager(tmp_path).summarize_shards())
    compacted = pd.read_parquet(summary)

    assert len(compacted) == 2
    assert set(compacted["ep_uid"]) == {"uid-a", "uid-b"}
    assert not first.exists()
    assert not second.exists()


def test_compaction_deduplicates_only_identical_episode_identity(tmp_path):
    row = _row(ep_uid="same-uid", ep_id=9, stamp=300, payload="same")
    _write_shard(tmp_path, 1, [row])
    _write_shard(tmp_path, 2, [row.copy()])

    summary = Path(_manager(tmp_path).summarize_shards())

    assert len(pd.read_parquet(summary)) == 1


def test_conflicting_uid_aborts_without_deleting_sources(tmp_path):
    first = _write_shard(
        tmp_path, 1, [_row(ep_uid="same-uid", ep_id=1, stamp=100, payload="a")]
    )
    second = _write_shard(
        tmp_path, 2, [_row(ep_uid="same-uid", ep_id=2, stamp=200, payload="b")]
    )
    manager = _manager(tmp_path)

    with pytest.raises(RuntimeError, match="Conflicting rows share episode identity"):
        manager.summarize_shards()

    assert first.exists()
    assert second.exists()
    assert not list((tmp_path / SCENARIO).glob("summary.*.parquet"))


def test_unreadable_shard_is_rejected_without_deleting_shards(tmp_path):
    readable = _write_shard(
        tmp_path, 1, [_row(ep_uid="uid-a", ep_id=1, stamp=100, payload="a")]
    )
    unreadable = tmp_path / SCENARIO / "episodes.000002.parquet"
    unreadable.write_bytes(b"not a parquet file")
    with pytest.raises(RuntimeError, match="Cannot read episode shard"):
        _manager(tmp_path)

    assert readable.exists()
    assert unreadable.exists()
    assert not list((tmp_path / SCENARIO).glob("summary.*.parquet"))


def test_failed_round_trip_verification_keeps_source_shards(tmp_path, monkeypatch):
    source = _write_shard(
        tmp_path, 1, [_row(ep_uid="uid-a", ep_id=1, stamp=100, payload="a")]
    )
    manager = _manager(tmp_path)
    real_read_parquet = pd.read_parquet

    def corrupt_summary_read(path, *args, **kwargs):
        frame = real_read_parquet(path, *args, **kwargs)
        if str(path).endswith(".tmp"):
            frame.loc[0, "payload_json"] = '{"value":"corrupt"}'
        return frame

    monkeypatch.setattr(pd, "read_parquet", corrupt_summary_read)

    with pytest.raises(RuntimeError, match="Could not write and verify summary"):
        manager.summarize_shards()

    assert source.exists()
    assert not list((tmp_path / SCENARIO).glob("summary.*.parquet"))
    assert not list((tmp_path / SCENARIO).glob("*.tmp"))


def test_new_uid_needs_no_coordination():
    from source.design_exploration.commons.episodes.episode_manager import new_episode_uid

    assert len({new_episode_uid() for _ in range(1000)}) == 1000


@pytest.mark.parametrize("legacy_cycle", [False, True])
def test_existing_shards_load_and_accept_new_rows_without_cycle(tmp_path, legacy_cycle):
    row = _row(ep_uid="existing", payload="existing")
    if legacy_cycle:
        row["cycle"] = 7
    original = _write_shard(tmp_path, 1, [row])
    before = original.read_bytes()
    manager = _manager(tmp_path)
    assert manager.episodes(0)[0]["payload"] == {"value": "existing"}
    new_uid = manager.append_one(
        0, {"value": "new"}, EpisodeMeta(scenario=SCENARIO, split="train", provenance="test")
    )
    manager.flush()
    new_shard = tmp_path / SCENARIO / "episodes.000002.parquet"
    assert "cycle" not in pd.read_parquet(new_shard).columns
    assert original.read_bytes() == before
    rows = _manager(tmp_path).episodes(0)
    assert [record["ep_uid"] for record in rows] == ["existing", new_uid]
    assert [record["payload"] for record in rows] == [{"value": "existing"}, {"value": "new"}]
