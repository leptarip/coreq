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

"""Campaign endpoint estimators: the three sampling schemes and the route families."""
import hashlib
import json
import math

import pytest

from source.design_exploration.rare_events.campaign_endpoints import (
    FAMILIES,
    PRINCIPAL_MODE,
    classify_family,
    estimate_family_vector,
    estimate_mode,
    mode_status,
    precision_status,
    stratified_estimate,
)


def row(*, failure, log_weight=0.0, detection=1000, age=50, component=None, signature=None):
    record = {
        "failure": failure,
        "log_weight": log_weight,
        "first_detection_ms": detection,
        "first_external_message_age_ms": age,
        "route_signature": signature if signature is not None else PRINCIPAL_MODE,
    }
    if component is not None:
        record["sampling_component"] = component
        record["proposal_log_ratios"] = {component: log_weight}
    return record


def test_iid_estimate_is_the_weighted_mean_of_the_hits():
    rows = [row(failure=True, log_weight=math.log(2.0)), row(failure=False), row(failure=False),
            row(failure=False)]
    hits = [r["failure"] for r in rows]
    out = estimate_mode("d", {"scheme": "iid_single_proposal"}, rows, hits)
    assert out["estimator"] == "iid_weighted"
    # One hit of weight 2 among four draws.
    assert out["estimate"] == pytest.approx(0.5)
    assert out["mode_events"] == 1


def test_nominal_monte_carlo_uses_the_exact_binomial():
    rows = [row(failure=i < 3) for i in range(100)]
    hits = [r["failure"] for r in rows]
    out = estimate_mode("d", {"scheme": "nominal_mc"}, rows, hits)
    assert out["estimator"] == "exact_binomial"
    assert out["estimate"] == pytest.approx(0.03)
    assert out["interval_method"] == "exact-clopper-pearson-two-sided"
    # An exact interval brackets the point estimate and a weighted ESS
    # does not apply to an unweighted sample.
    assert out["ci_lower"] < out["estimate"] < out["ci_upper"]
    assert out["mode_ess"] is None and out["max_event_contribution"] is None


def test_stratified_estimate_pools_within_strata():
    rows = ([row(failure=True, log_weight=math.log(2.0), component="a")] +
            [row(failure=False, component="a")] * 3 +
            [row(failure=True, log_weight=math.log(4.0), component="b")] +
            [row(failure=False, component="b")] * 3)
    out = stratified_estimate(rows, indicator=[r["failure"] for r in rows])
    assert out["estimate"] == pytest.approx((2.0 + 4.0) / 8)
    assert out["stratum_counts"] == {"a": 4, "b": 4}
    assert out["interval_method"].startswith("stratified-")


def test_stratified_estimation_refuses_an_iid_corpus():
    rows = [row(failure=True), row(failure=False)]
    with pytest.raises(ValueError, match="sampling_component"):
        stratified_estimate(rows, indicator=[True, False])


def test_every_stratum_needs_two_samples():
    rows = [row(failure=True, component="a"), row(failure=False, component="b")]
    with pytest.raises(ValueError, match="at least two samples"):
        stratified_estimate(rows, indicator=[True, False])


def test_route_families_partition_the_failures():
    assert classify_family(row(failure=True, detection=5000)) == "late_detection"
    assert classify_family(row(failure=True, detection=100, age=5000)) == "early_detection_stale_first_delivery"
    assert classify_family(row(failure=True, detection=100, age=10)) == "early_detection_fresh_first_delivery"
    # A safe episode belongs to no failure family.
    assert classify_family(row(failure=False)) is None


def test_family_vector_sums_to_the_aggregate():
    rows = [row(failure=True, detection=5000), row(failure=True, detection=100, age=5000),
            row(failure=True, detection=100, age=10), row(failure=False)]
    vector = estimate_family_vector(rows, "iid_is")
    total = math.fsum(vector["family_estimates"][f]["estimate"] for f in FAMILIES)
    assert vector["aggregate"]["estimate"] == pytest.approx(total)
    assert vector["aggregate"]["partition_residual"] == pytest.approx(0.0, abs=1e-15)


def test_precision_gates_mark_a_thin_estimate_as_exploratory():
    thin = {"events_under_proposal": 2, "event_ess": 1.5, "maximum_event_contribution": 0.9,
            "relative_standard_error": 0.8}
    status = precision_status(thin, "iid_is")
    assert status["status"] == "imprecise_exploratory"
    assert set(status["failed_gates"]) == {
        "minimum_event_paths", "minimum_event_ess",
        "maximum_event_contribution", "maximum_relative_standard_error",
    }


def test_zero_events_is_not_evidence_of_zero_risk():
    status = mode_status("iid_single_proposal", {"mode_events": 0})
    assert status["status"] == "not_observed"
    assert "zero nominal risk" in status["reason"]


def test_unknown_scheme_is_rejected():
    with pytest.raises(ValueError, match="unknown sampling scheme"):
        estimate_mode("d", {"scheme": "made_up"}, [row(failure=False)], [False])


# --- the replay driver -------------------------------------------------------

def write_campaign(tmp_path, *, corrupt=False):
    """A minimal campaign directory: one job, one shard, one digest sidecar."""
    rows = [row(failure=i < 2, log_weight=math.log(2.0) if i < 2 else 0.0) for i in range(10)]
    job_dir = tmp_path / "estimate-x" / "sample-shards"
    job_dir.mkdir(parents=True)
    shard = job_dir / "000000.jsonl"
    shard.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    if corrupt:
        digest = "0" * 64
    (job_dir / "000000.sha256.json").write_text(json.dumps({"sha256": digest}))
    (tmp_path / "seed-plan.json").write_text(json.dumps(
        [{"key": "estimate-x", "scheme": "iid_single_proposal", "design": 1, "episodes": len(rows)}]))
    return tmp_path


def test_replay_recomputes_the_endpoints(tmp_path):
    from reproduction.phase3 import replay_campaign
    out = replay_campaign.replay(write_campaign(tmp_path))
    mode = out["jobs"]["estimate-x"]["mode"]
    assert mode["estimate"] == pytest.approx(4.0 / 10)
    assert out["jobs"]["estimate-x"]["episodes"] == 10


def test_replay_rejects_a_tampered_shard(tmp_path):
    from reproduction.phase3 import replay_campaign
    with pytest.raises(ValueError, match="digest mismatch"):
        replay_campaign.replay(write_campaign(tmp_path, corrupt=True))


def test_verify_reports_a_differing_endpoint(tmp_path):
    from reproduction.phase3 import replay_campaign
    replayed = replay_campaign.replay(write_campaign(tmp_path))
    published = {"M1": {"estimate-x": {"estimate": 0.123, "estimator": "iid_weighted"}}}
    failures = replay_campaign.verify(replayed, published)
    assert len(failures) == 1 and "M1[estimate-x].estimate" in failures[0]
    # The same comparison passes against the value it actually computed.
    published["M1"]["estimate-x"]["estimate"] = replayed["jobs"]["estimate-x"]["mode"]["estimate"]
    assert replay_campaign.verify(replayed, published) == []
