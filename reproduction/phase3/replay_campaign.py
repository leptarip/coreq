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
Recompute a rare-event campaign's reported endpoints from its sample shards.

A campaign stores one JSON record per episode in `sample-shards`, each carrying
the likelihood ratio, the failure indicator and the route features. The reported
probabilities are a weighted aggregation of those records, so they can be
recomputed here without a simulator.

What this does not do is re-derive the shard contents themselves. Route
signatures and likelihood ratios come from the simulator logs, and checking them
is the campaign's own audit, which needs the trajectories. The shards are taken
as given, and their integrity is checked against the sha256 sidecar written
beside each one.

    python -m reproduction.phase3.replay_campaign \\
        --campaign <dir> --published results/phase3/<name>.json

Exits non-zero when a recomputed endpoint differs from the published one.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

_HERE = Path(__file__).resolve()
for _cand in [_HERE] + list(_HERE.parents):
    if (_cand / "source").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break

from source.design_exploration.rare_events.campaign_endpoints import (  # noqa: E402
    FAMILIES,
    PRINCIPAL_MODE,
    estimate_family_vector,
    estimate_mode,
    mode_status,
    precision_status,
)

# The scheme a job was run under decides which estimator applies.
FAMILY_SCHEME = {
    "iid_single_proposal": "iid_is",
    "deterministic_mixture": "stratified_is",
    "nominal_mc": "nominal_mc",
}
# Endpoint fields worth comparing against a published file.
COMPARED_FIELDS = (
    "estimator", "estimate", "standard_error", "relative_standard_error",
    "ci_lower", "ci_upper", "mode_events", "aggregate_failures",
    "interval_method", "one_sided_upper", "status", "reason",
)
AGGREGATE_FIELDS = (
    "estimate", "standard_error", "relative_standard_error", "partition_residual",
)
TOLERANCE = 1e-12


def load_shards(campaign: Path, job_key: str) -> Tuple[List[Dict[str, Any]], int]:
    """Read one job's rows in planned order, verifying each shard's digest."""
    paths = sorted(
        glob.glob(str(campaign / job_key / "sample-shards" / "*.jsonl")),
        key=lambda p: int(os.path.basename(p).split(".")[0]),
    )
    if not paths:
        raise FileNotFoundError(f"No sample shards for job {job_key!r} under {campaign}")
    rows: List[Dict[str, Any]] = []
    for path in paths:
        digest_path = Path(path).with_suffix(".sha256.json")
        if not digest_path.is_file():
            raise FileNotFoundError(f"Missing digest sidecar for {path}")
        recorded = json.loads(digest_path.read_text())["sha256"]
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if recorded != actual:
            raise ValueError(f"Shard digest mismatch: {path}")
        rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    return rows, len(paths)


def endpoints_for_job(job: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The principal mode and the route-family vector for one job's rows."""
    hits = [bool(row["failure"]) and row["route_signature"] == PRINCIPAL_MODE for row in rows]
    mode = estimate_mode(str(job["design"]), job, rows, hits)
    # The precision gates read the event count and ESS under their reporting
    # names; the estimator returns them under its own.
    mode.update(
        events_under_proposal=mode["mode_events"],
        event_ess=mode["mode_ess"],
        maximum_event_contribution=mode["max_event_contribution"],
    )
    mode["precision_diagnostics"] = precision_status(mode, FAMILY_SCHEME[job["scheme"]])
    mode.update(mode_status(job["scheme"], mode))
    vector = estimate_family_vector(rows, FAMILY_SCHEME[job["scheme"]])
    for record in vector["family_estimates"].values():
        record["precision_diagnostics"] = precision_status(record, FAMILY_SCHEME[job["scheme"]])
    return {"mode": mode, "families": vector}


def replay(campaign: Path) -> Dict[str, Any]:
    """Recompute every job in a campaign's seed plan."""
    jobs = json.loads((campaign / "seed-plan.json").read_text())
    out: Dict[str, Any] = {"campaign": campaign.name, "jobs": {}}
    for job in jobs:
        key = job["key"]
        if not (campaign / key / "sample-shards").is_dir():
            continue
        rows, shard_count = load_shards(campaign, key)
        if len(rows) != job["episodes"]:
            raise ValueError(
                f"{key}: {len(rows)} rows against {job['episodes']} planned episodes."
            )
        result = endpoints_for_job(job, rows)
        result.update(scheme=job["scheme"], episodes=len(rows), shards=shard_count)
        out["jobs"][key] = result
    if not out["jobs"]:
        raise FileNotFoundError(f"No job shards found under {campaign}")
    return out


def _same(got: Any, want: Any) -> bool:
    if isinstance(want, str) or isinstance(want, bool):
        return got == want
    if want is None:
        return got is None
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return math.isclose(float(got), float(want), rel_tol=TOLERANCE, abs_tol=1e-25)
    return got == want


def _compare(label: str, got: Mapping[str, Any], want: Mapping[str, Any],
             fields: Sequence[str], failures: List[str]) -> None:
    for field in fields:
        if field not in want:
            continue
        if not _same(got.get(field), want[field]):
            failures.append(f"{label}.{field}: recomputed {got.get(field)!r} != published {want[field]!r}")


def verify(replayed: Mapping[str, Any], published: Mapping[str, Any]) -> List[str]:
    """Compare against a published analysis file, in either of its two shapes."""
    failures: List[str] = []
    jobs = replayed["jobs"]
    if "estimates" in published:
        # Single-job campaign: the mode and the families share one block.
        (key, result), = jobs.items()
        _compare("M1", result["mode"], published["estimates"]["M1"], COMPARED_FIELDS, failures)
        for family in FAMILIES:
            if family in published["estimates"]:
                _compare(family, result["families"]["family_estimates"][family],
                         published["estimates"][family], COMPARED_FIELDS, failures)
        if "aggregate" in published:
            _compare("aggregate", result["families"]["aggregate"], published["aggregate"],
                     AGGREGATE_FIELDS, failures)
    else:
        # Multi-job campaign: one mode per job, plus per-job family evidence.
        for key, want in published.get("M1", {}).items():
            if key not in jobs:
                failures.append(f"{key}: published but not replayed")
                continue
            _compare(f"M1[{key}]", jobs[key]["mode"], want, COMPARED_FIELDS, failures)
        for key, want in published.get("family_probability_evidence", {}).items():
            if key in jobs and "aggregate" in want:
                _compare(f"families[{key}]", jobs[key]["families"]["aggregate"],
                         want["aggregate"], AGGREGATE_FIELDS, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--campaign", required=True, type=Path,
                        help="Campaign directory holding seed-plan.json and the job shards.")
    parser.add_argument("--published", type=Path,
                        help="Published analysis JSON to verify the replay against.")
    parser.add_argument("--out", type=Path, help="Write the recomputed endpoints here.")
    args = parser.parse_args()

    replayed = replay(args.campaign)
    for key, result in replayed["jobs"].items():
        mode = result["mode"]
        print(f"[{key}] scheme={result['scheme']} episodes={result['episodes']} "
              f"shards={result['shards']} | {mode['estimator']} estimate={mode['estimate']:.6g} "
              f"CI=[{mode['ci_lower']:.3g}, {mode['ci_upper']:.3g}] events={mode['mode_events']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(replayed, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    if args.published:
        failures = verify(replayed, json.loads(args.published.read_text()))
        if failures:
            print(f"\n{len(failures)} endpoint(s) differ from {args.published}:")
            for line in failures:
                print(f"  {line}")
            return 1
        print(f"\nevery endpoint in {args.published.name} reproduced from the shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
