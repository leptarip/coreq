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

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# Ensure the repository root (folder containing "source") is on sys.path when run as a script.
_HERE = Path(__file__).resolve()
for cand in [_HERE] + list(_HERE.parents):
    if (cand / "source").is_dir():
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
        break

from source.design_exploration.commons.design_space import SPACE_SPEC
from source.design_exploration.commons.episodes.episode_manager import EpisodeManager
from source.design_exploration.commons.episodes.episode_result import EpisodeResult
from source.design_exploration.commons.scenario_labeling import (
    SPRTConfig,
    SPRTState,
    TestResult,
    required_min_iters_k0,
    sprt_decision_for_counts,
)

_REPO_ROOT = next((cand for cand in [_HERE] + list(_HERE.parents) if (cand / "source").is_dir()), _HERE.parent)
DEFAULT_EPISODE_ROOT = str(_REPO_ROOT / "episode_dataset")
DEFAULT_SCENARIOS = ("intersection1", "intersection2")
DEFAULT_SEED = 13
# IDE-friendly knobs; tweak these when running the script interactively.
USE_SCENARIOS = list(DEFAULT_SCENARIOS)
USE_EPISODE_ROOT = DEFAULT_EPISODE_ROOT


def _payload_to_episode_result(payload: Dict) -> EpisodeResult:
    ep = EpisodeResult()
    ep.result = payload if isinstance(payload, dict) else {"payload": payload}
    return ep


def make_min_dist_violation_function(d_safe: float) -> Callable[[EpisodeResult], TestResult]:
    threshold = float(d_safe)

    def _min_dist_violation_function(ep_result: EpisodeResult) -> TestResult:
        result = getattr(ep_result, "result", {}) or {}
        safety = result.get("safety_m", {}) or {}
        try:
            min_d = float(safety.get("min_d"))
        except (TypeError, ValueError):
            print(f"Warning: could not parse min_d from episode result: {safety}", file=sys.stderr)
            return TestResult.UNSAFE
        return TestResult.SAFE if min_d >= threshold else TestResult.UNSAFE

    return _min_dist_violation_function


min_dist_violation_function = make_min_dist_violation_function(0.5)


@dataclass
class LabelOutcome:
    label: Optional[TestResult]
    mode: str
    episodes_used: int
    failures: int
    log_likelihood: float = 0.0


def offline_sprt_label(
    episodes: Iterable[EpisodeResult],
    cfg: SPRTConfig,
    violation_fn: Callable[[EpisodeResult], TestResult],
) -> LabelOutcome:
    """
    Run the SPRT/CP logic offline on a finite set of cached episodes.
    Returns label=None when we exhaust the cache without reaching any stopping rule.
    """
    state = SPRTState(cfg)

    for ep in episodes:
        vio = violation_fn(ep)
        decision = state.observe(vio)
        if decision is not None:
            return LabelOutcome(
                decision.label,
                decision.mode,
                state.n,
                state.k,
                state.log_likelihood,
            )

    return LabelOutcome(
        None,
        "INSUFFICIENT_EPISODES",
        state.n,
        state.k,
        state.log_likelihood,
    )


def min_additional_eps_to_decide(n: int, k: int, cfg: SPRTConfig) -> int:
    """
    Lower bound on how many more episodes are needed to reach any stopping rule
    given the current stats (optimistic: assumes future episodes are all SAFE or all UNSAFE).
    """
    if sprt_decision_for_counts(n, k, cfg) is not None:
        return 0

    alpha_err = 1.0 - cfg.C
    p0 = cfg.P
    p1 = cfg.p1_factor * cfg.P
    logA = math.log((1.0 - cfg.beta_err) / alpha_err)
    logB = math.log(cfg.beta_err / (1.0 - alpha_err))
    min_iters_k0 = required_min_iters_k0(cfg.P, alpha_err)
    safe_inc = math.log((1.0 - p1) / (1.0 - p0))
    unsafe_inc = math.log(p1 / p0)
    logL = k * unsafe_inc + (n - k) * safe_inc

    candidates: List[int] = []
    if k == 0 and n < min_iters_k0:
        candidates.append(min_iters_k0 - n)
    if safe_inc > 0 and logL < logA:
        candidates.append(int(math.ceil((logA - logL) / safe_inc)))
    if unsafe_inc < 0 and logL > logB:
        candidates.append(int(math.ceil((logL - logB) / abs(unsafe_inc))))
    if n < cfg.global_max_iters:
        candidates.append(cfg.global_max_iters - n)
    if not candidates:
        return 0
    need = max(0, min(candidates))
    return need



def build_episode_manager(episode_root: str, scenario: str) -> EpisodeManager:
    space_spec = SPACE_SPEC
    grid = space_spec.build_grid()
    ep_root = os.path.join(episode_root, scenario)
    if not os.path.isdir(ep_root):
        raise FileNotFoundError(f"Episode directory not found: {ep_root}")
    return EpisodeManager(root_dir=ep_root, scenario_name=scenario, space_spec=space_spec, GRID=grid)


def label_scenario(
    epman: EpisodeManager,
    cfg: SPRTConfig,
    violation_fn: Callable[[EpisodeResult], TestResult],
    eval_idx: Iterable[int],
) -> Tuple[int, int, int, int, int, int, int, Optional[float], Optional[float]]:
    """
    Returns (total_designs_with_eps, labelled_designs, insufficient_designs,
             eval_total, eval_labelled, eval_missing, eval_missing_eps_est).
    """
    counts = epman.list_gidx_with_counts()
    labelled = 0
    insufficient = 0
    outcomes: Dict[int, LabelOutcome] = {}
    min_d_vals_unsafe: List[float] = []

    for gidx, _ in sorted(counts.items()):
        cached = epman.episodes(gidx, consume=False)
        if not cached:
            insufficient += 1
            continue
        eps = [_payload_to_episode_result(ep.get("payload", {})) for ep in cached]
        outcome = offline_sprt_label(eps, cfg, violation_fn)
        outcomes[int(gidx)] = outcome
        if outcome.label is None:
            insufficient += 1
        else:
            labelled += 1
            if outcome.label == TestResult.UNSAFE:
                for ep in eps:
                    safety = getattr(ep, "result", {}).get("safety_m", {}) or {}
                    try:
                        md = float(safety.get("min_d"))
                        min_d_vals_unsafe.append(md)
                    except Exception:
                        continue

    eval_labelled = 0
    eval_missing = 0
    eval_missing_eps_est = 0
    eval_list = list(eval_idx)
    eval_total = len(eval_list)
    for gi in eval_list:
        outcome = outcomes.get(int(gi))
        if outcome is None:
            eval_missing += 1  # no episodes for this design
            eval_missing_eps_est += cfg.global_max_iters
        elif outcome.label is None:
            eval_missing += 1
            eval_missing_eps_est += min_additional_eps_to_decide(outcome.episodes_used, outcome.failures, cfg)
        else:
            eval_labelled += 1

    min_md = min(min_d_vals_unsafe) if min_d_vals_unsafe else None
    max_md = max(min_d_vals_unsafe) if min_d_vals_unsafe else None
    return len(counts), labelled, insufficient, eval_total, eval_labelled, eval_missing, eval_missing_eps_est, min_md, max_md


def print_summary(rows: List[Tuple[str, int, int, int, int, int, int, int, Optional[float], Optional[float]]]) -> None:
    if not rows:
        print("No scenarios processed.")
        return
    header = (
        f"{'scenario':<16}"
        f"{'designs':>12}"
        f"{'labelled':>12}"
        f"{'insufficient':>15}"
        f"{'grid_total':>15}"
        f"{'grid_labelled':>17}"
        f"{'grid_missing':>16}"
        f"{'miss_eps_est':>15}"
        f"{'min_d_unsafe':>15}"
        f"{'max_d_unsafe':>15}"
    )
    print("\n" + header)
    print("-" * len(header))
    for scenario, total, labelled, insufficient, audit_total, audit_labelled, audit_missing, miss_eps, min_md, max_md in rows:
        print(
            f"{scenario:<16}"
            f"{total:>12}"
            f"{labelled:>12}"
            f"{insufficient:>15}"
            f"{audit_total:>15}"
            f"{audit_labelled:>17}"
            f"{audit_missing:>16}"
            f"{miss_eps:>15}"
            f"{(f'{min_md:.4f}' if min_md is not None else 'n/a'):>15}"
            f"{(f'{max_md:.4f}' if max_md is not None else 'n/a'):>15}"
        )


def main() -> None:
    cfg = SPRTConfig()
    rows: List[Tuple[str, int, int, int, int, int, int, int]] = []

    episode_root = USE_EPISODE_ROOT
    scenarios = USE_SCENARIOS
    eval_idx = range(len(SPACE_SPEC.build_grid()))

    for scenario in scenarios:
        try:
            epman = build_episode_manager(episode_root, scenario)
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
            continue

        (
            total,
            labelled,
            insufficient,
            audit_total,
            audit_labelled,
            audit_missing,
            audit_missing_eps,
            min_md,
            max_md,
        ) = label_scenario(epman, cfg, min_dist_violation_function, eval_idx)
        print(
            f"{scenario}: designs_with_eps={total}, labelled={labelled}, insufficient={insufficient}, "
            f"audit_total={audit_total}, audit_labelled={audit_labelled}, audit_missing={audit_missing}, "
            f"audit_missing_eps_est={audit_missing_eps}"
        )
        rows.append(
            (
                scenario,
                total,
                labelled,
                insufficient,
                audit_total,
                audit_labelled,
                audit_missing,
                audit_missing_eps,
                min_md,
                max_md,
            )
        )

    print_summary(rows)


if __name__ == "__main__":
    main()
