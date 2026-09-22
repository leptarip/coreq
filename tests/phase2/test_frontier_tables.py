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

"""The published frontier tables against the held-out episodes behind them.

Every performance number the figure draws is a mean over the held-out block:
trials 61-300 of each design, disjoint from the 60 that selected it. These
tests re-derive each published mean from `heldout_episode_values.csv`, so a
table that drifts from its evidence fails here rather than in a figure nobody
regenerates.
"""
import csv
from collections import defaultdict
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[2] / "results" / "phase2" / "frontier"
TOLERANCE = 1e-9
SELECTION_TRIALS = 60
HELDOUT_TRIALS = 240


def rows(name):
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def episodes():
    """Held-out episodes per design, as (metric, min_d) value lists."""
    metric, min_d = defaultdict(list), defaultdict(list)
    ordinals = defaultdict(list)
    for row in rows("heldout_episode_values.csv"):
        design = int(row["design_id"])
        metric[design].append(float(row["paired_metric"]))
        min_d[design].append(float(row["paired_min_d"]))
        ordinals[design].append(int(row["seed_ordinal"]))
    return {"metric": metric, "min_d": min_d, "ordinals": ordinals}


def mean(values):
    return sum(values) / len(values)


def test_every_design_has_the_full_heldout_block(episodes):
    for design, ordinals in episodes["ordinals"].items():
        # The block is exactly trials 61-300: the selection trials are excluded,
        # so the reported mean cannot inherit the selection advantage.
        assert sorted(ordinals) == list(
            range(SELECTION_TRIALS + 1, SELECTION_TRIALS + HELDOUT_TRIALS + 1)
        ), f"design {design} has an incomplete held-out block"


def test_baseline_frontier_means_come_from_the_episodes(episodes):
    for row in rows("baseline_frontier.csv"):
        design = int(row["design_id"])
        published = float(row["heldout_r240_metric_mean"])
        assert abs(published - mean(episodes["metric"][design])) < TOLERANCE, design


def test_combined_frontier_min_d_comes_from_the_episodes(episodes):
    # The figure reads this column instead of the episode file; it must still
    # be the mean of those episodes.
    for row in rows("combined_safe_frontier.csv"):
        design = int(row["design_id"])
        published = float(row["mean_min_d"])
        assert abs(published - mean(episodes["min_d"][design])) < TOLERANCE, design


def test_stage_c_means_come_from_the_episodes(episodes):
    checked = 0
    for row in rows("stage_c_final.csv"):
        design = int(row["design_id"])
        if design not in episodes["metric"]:
            continue
        published = float(row["heldout_r240_metric_mean"])
        assert abs(published - mean(episodes["metric"][design])) < TOLERANCE, design
        checked += 1
    assert checked == 11, "expected 11 stage-C designs covered by the episode file"


def test_challenger_means_come_from_the_episodes(episodes):
    checked = 0
    for row in rows("performance_r300_final.csv"):
        design = int(row["design_id"])
        if design not in episodes["metric"]:
            continue
        published = float(row["candidate_heldout_metric_mean"])
        assert abs(published - mean(episodes["metric"][design])) < TOLERANCE, design
        checked += 1
    assert checked == 2, "expected the two challengers, 6499 and 7310"


def test_the_two_frontiers_agree_where_they_overlap():
    baseline = {int(r["design_id"]): float(r["heldout_r240_metric_mean"])
                for r in rows("baseline_frontier.csv")}
    for row in rows("combined_safe_frontier.csv"):
        design = int(row["design_id"])
        if design in baseline:
            assert abs(float(row["metric"]) - baseline[design]) < TOLERANCE, design


def test_the_extended_frontier_is_what_the_figure_expects():
    from reproduction.phase2.plot_frontier import (
        EXPECTED_AUGMENTED_IDS, EXPECTED_BASELINE_IDS, EXPECTED_NEW_IDS,
    )
    baseline = tuple(int(r["design_id"]) for r in rows("baseline_frontier.csv"))
    augmented = tuple(int(r["design_id"]) for r in rows("combined_safe_frontier.csv"))
    assert baseline == EXPECTED_BASELINE_IDS
    assert augmented == EXPECTED_AUGMENTED_IDS
    # The challengers are in the extended frontier and absent from the baseline.
    assert set(EXPECTED_NEW_IDS) <= set(augmented)
    assert not set(EXPECTED_NEW_IDS) & set(baseline)
