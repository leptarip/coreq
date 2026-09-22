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
# rf_active_learning.py — cache-first AL with calibrated RF, adaptive boundary picking,
# stopping criteria, and detailed trackers (merged from approach1_tracked.py)
from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV

import logging

# -----------------------
# Logger
# -----------------------
logger = logging.getLogger(__name__)

from source.design_exploration.commons import design_space
from source.design_exploration.commons.scenario_labeling import ScenarioTester, SPRTConfig
from source.design_exploration.commons.episodes.episode_manager import EpisodeManager, EpisodeMeta
from source.design_exploration.commons.episodes.merged_episode_manager import build_merged_episode_manager
from source.design_exploration.commons.episodes.episode_result import EpisodeResult
from source.design_exploration.commons.offline_labeling import offline_sprt_label


# =======================
# Config
# =======================
@dataclass
class ActiveCfg:
    # core
    seed: int = 0
    init_k: int = 32
    batch: int = 16
    iters: int = 40
    flush_every: int = 100
    out_dir: str = "learner_artifacts"

    # calibration
    calibrate: bool = True
    calib_method: str = "sigmoid"   # or "isotonic"
    calib_cv: int = 5

    # boundary picking (from tracked version)
    t_prov_method: str = "quantile"  # "quantile" | "uncertainty"
    t_prov_param: float = 0.8        # quantile q if method="quantile"
    adaptive_quantile: bool = True   # adapt q via previous acceptance mass
    quantile_band_width: float = 0.05  # epsilon for "boundary mass" & candidate band

    # stopping
    early_stop_jaccard_eps: float = 0.002   # stop if masks hardly change…
    early_stop_patience: int = 3            # …for this many consecutive cycles (0 = disabled)


# =======================
# Models
# =======================
def build_rf(seed: int = 0):
    return RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
    )

def _fit_model(grid, idx, y, seed, cfg: ActiveCfg):
    yy = np.array(y, dtype=int)
    if len(idx) == 0 or len(np.unique(yy)) < 2:
        clf = DummyClassifier(strategy="prior")
        X = grid[idx] if len(idx) else np.zeros((0, grid.shape[1]), dtype=float)
        clf.fit(X, yy if len(idx) else np.zeros((0,), dtype=int))
        return clf

    base = build_rf(seed=seed)
    if cfg.calibrate:
        try:
            model = CalibratedClassifierCV(base_estimator=base,
                                           method=cfg.calib_method,
                                           cv=cfg.calib_cv)
            model.fit(grid[idx], yy)
            return model
        except Exception:
            logger.exception("Calibration failed (method=%s, cv=%s); falling back to uncalibrated RF",
                             cfg.calib_method, cfg.calib_cv)
    base.fit(grid[idx], yy)
    return base


# =======================
# Utilities (selection / boundary / tracking)
# =======================
def farthest_first(X: np.ndarray, k: int, seed: Optional[int] = None) -> np.ndarray:
    n = len(X)
    if n == 0 or k <= 0:
        return np.array([], dtype=int)
    rng = np.random.RandomState(seed if seed is not None else None)
    start = int(rng.randint(0, n))
    sel = [start]
    d2 = np.sum((X - X[start]) ** 2, axis=1)
    for _ in range(1, min(k, n)):
        i = int(np.argmax(d2))
        sel.append(i)
        d2 = np.minimum(d2, np.sum((X - X[i]) ** 2, axis=1))
    return np.array(sel, dtype=int)

def choose_t_prov(scores_train: np.ndarray,
                  method: str = "quantile",
                  param: float = 0.8,
                  t_last: Optional[float] = None) -> float:
    s = np.clip(scores_train.astype(float), 0.0, 1.0)
    if method == "quantile":
        q = float(min(max(param, 0.0), 1.0))
        return float(np.quantile(s, q))
    elif method == "uncertainty":
        return float(0.5 if param is None else float(param))
    return float(np.quantile(s, 0.5))

def pick_batch_risk_aware(grid_X: np.ndarray, scores: np.ndarray, t_prov: float,
                          already_labeled_mask: np.ndarray, batch_size: int,
                          shortlist_factor: int = 20, seed: int = 0,
                          band_width: Optional[float] = None) -> np.ndarray:
    unlabeled = np.where(~already_labeled_mask)[0]
    if len(unlabeled) == 0:
        return np.array([], dtype=int)

    margin = np.abs(scores - t_prov)
    if band_width is not None and band_width > 0:
        in_band = unlabeled[margin[unlabeled] <= band_width]
        pool = in_band if len(in_band) >= batch_size else unlabeled
    else:
        pool = unlabeled

    prio = 1.0 / (margin[pool] + 1e-6)
    R = min(len(pool), max(batch_size * shortlist_factor, batch_size))
    shortlist = pool[np.argsort(prio)[::-1][:R]]
    rel = farthest_first(grid_X[shortlist], k=batch_size, seed=seed)
    return shortlist[rel]

def _jaccard(prev_mask: Optional[np.ndarray], curr_mask: np.ndarray) -> float:
    if prev_mask is None:
        return 1.0
    inter = np.logical_and(prev_mask, curr_mask).sum()
    union = np.logical_or(prev_mask, curr_mask).sum()
    if union == 0:
        return 0.0
    return 1.0 - (inter / union)

def _payload_to_episode_result(payload_dict) -> EpisodeResult:
    ep = EpisodeResult()
    ep.result = payload_dict if isinstance(payload_dict, dict) else {"payload": payload_dict}
    return ep


# =======================
# Cache-first labeling
# =======================
def _label_via_cache_or_sim(gi: int,
                            design: design_space.Design,
                            tester: ScenarioTester,
                            epman: EpisodeManager,
                            scenario_name: str) -> Tuple[int, int, str]:
    """
    Returns (label, sim_episode_count, mode)
    - Uses all cached episodes with offline SPRT; if insufficient, simulates batches
      until a label is reached (or the SPRT cap is hit), then logs new episodes.
    """
    sprt_cfg = getattr(tester, "sprt_cfg", None) or SPRTConfig(
        P=tester.P,
        C=1.0 - tester.alpha_err,
        p1_factor=tester.p1 / tester.p0 if tester.p0 else SPRTConfig.p1_factor,
        beta_err=tester.beta_err,
        global_max_iters=tester.global_max_iters,
        batch_size=tester.batch_size,
    )

    cached_rows = epman.episodes(gi, consume=False)
    eps: List[EpisodeResult] = [_payload_to_episode_result(r["payload"]) for r in cached_rows]
    outcome = offline_sprt_label(eps, sprt_cfg, tester.violation_function)
    if outcome.label is not None:
        logger.info(
            "AL label design gidx=%d | decided from cache with total_eps=%d | mode=%s | label=%d",
            int(gi),
            int(len(eps)),
            str(outcome.mode),
            int(outcome.label),
        )

    simulated: List[EpisodeResult] = []
    while outcome.label is None:
        remaining = sprt_cfg.global_max_iters - len(eps)
        if remaining <= 0:
            break
        batch_n = min(sprt_cfg.batch_size, remaining)
        logger.info(
            "AL label design gidx=%d | simulating batch=%d | cached_eps=%d | simulated_so_far=%d",
            int(gi),
            int(batch_n),
            int(len(eps)),
            int(len(simulated)),
        )
        new_eps = tester._episodes(design, batch_n)
        simulated.extend(new_eps)
        eps.extend(new_eps)
        outcome = offline_sprt_label(eps, sprt_cfg, tester.violation_function)
        if outcome.label is not None:
            logger.info(
                "AL label design gidx=%d | decided after total_eps=%d | new_sim_eps=%d | mode=%s | label=%d",
                int(gi),
                int(len(eps)),
                int(len(simulated)),
                str(outcome.mode),
                int(outcome.label),
            )
        if len(eps) >= sprt_cfg.global_max_iters:
            break

    if simulated:
        epman.append_batch(
            gi,
            simulated,
            EpisodeMeta(scenario=scenario_name, split="train", provenance="AL-RF")
        )
        epman.flush()

    if outcome.label is None:
        raise RuntimeError(
            f"SPRT still inconclusive at n={len(eps)} (max={sprt_cfg.global_max_iters})."
        )

    final_label = outcome.label
    return int(final_label), len(simulated), str(outcome.mode)


# =======================
# Main loop
# =======================
def active_grid_learning(learner_dir: str,
                         scenario_name: str,
                         tester: ScenarioTester,
                         act: ActiveCfg,
                         GRID: np.ndarray,
                         train_idx: np.ndarray,
                         space_spec: design_space.SpaceSpec,
                         algo_version: str,
                         episode_cache_roots: Optional[Sequence[str]] = None) -> Dict:
    # Normalize learner_dir so outputs match other surrogates: <root>/<scenario>/...
    base_dir = learner_dir
    scenario_dir = (
        base_dir
        if os.path.basename(os.path.normpath(base_dir)) == scenario_name
        else os.path.join(base_dir, scenario_name)
    )
    os.makedirs(scenario_dir, exist_ok=True)

    logger.info(
        "AL start | scenario=%s | GRID=%d | train=%d | init_k=%d | batch=%d | iters=%d | calibrate=%s",
        scenario_name, len(GRID), len(train_idx), act.init_k, act.batch, act.iters, str(act.calibrate)
    )

    ep_dir = os.path.join(scenario_dir, "episodes", scenario_name)
    os.makedirs(ep_dir, exist_ok=True)
    logger.info("Episode cache: writing new episodes to %s", ep_dir)

    epman = build_merged_episode_manager(
        write_root_dir=ep_dir,
        scenario_name=scenario_name,
        space_spec=space_spec,
        GRID=GRID,
        flush_every=act.flush_every,
        episode_cache_roots=episode_cache_roots,
    )
    logger.info("Episode cache: reading cached episodes from %d source(s)", len(epman.source_dirs))
    for idx, path in enumerate(epman.source_dirs, start=1):
        logger.info("Episode cache source %d/%d | %s", idx, len(epman.source_dirs), path)
    counts = epman.list_gidx_with_counts()
    if counts:
        total_cached = int(sum(counts.values()))
        designs_with_cache = int(len(counts))
        total_unconsumed = int(sum(epman.count(gi, unconsumed_only=True) for gi in counts))
    else:
        total_cached = designs_with_cache = total_unconsumed = 0

    logger.info(
        "Episode cache | scenario=%s | designs_with_eps=%d | total_eps=%d | unconsumed_eps=%d",
        scenario_name, designs_with_cache, total_cached, total_unconsumed
    )

    # Warm start from cached episodes: label any train design with enough cached data.
    sprt_cfg = getattr(tester, "sprt_cfg", None) or SPRTConfig(
        P=tester.P,
        C=1.0 - tester.alpha_err,
        p1_factor=tester.p1 / tester.p0 if tester.p0 else SPRTConfig.p1_factor,
        beta_err=tester.beta_err,
        global_max_iters=tester.global_max_iters,
        batch_size=tester.batch_size,
    )
    warmup_labeled: List[int] = []
    warmup_labels: List[int] = []
    warmup_unlabeled = 0
    for gi in train_idx:
        if counts.get(int(gi), 0) <= 0:
            continue
        rows = epman.episodes(int(gi), consume=False)
        if not rows:
            continue
        eps = [_payload_to_episode_result(r.get("payload", {})) for r in rows]
        outcome = offline_sprt_label(eps, sprt_cfg, tester.violation_function)
        if outcome.label is None:
            warmup_unlabeled += 1
            continue
        warmup_labeled.append(int(gi))
        warmup_labels.append(int(outcome.label))

    if warmup_labeled or warmup_unlabeled:
        logger.info(
            "Cache warmup | labeled=%d | insufficient=%d",
            len(warmup_labeled),
            warmup_unlabeled,
        )
    else:
        logger.info("Cache warmup | no cached designs labeled")

    # Initial labeled pool
    L: List[int] = list(warmup_labeled)
    y: List[int] = list(warmup_labels)

    total_episodes_spent = 0

    # Seed via cache-or-sim if no cached labels were usable
    if len(L) == 0:
        grid01, _, _ = design_space.normalize_grid(GRID[train_idx])
        init_sel_local = farthest_first(grid01, k=min(act.init_k, len(train_idx)), seed=act.seed)
        init_seed = list(train_idx[init_sel_local])
        seed_total = len(init_seed)
        logger.info("AL init seeding will label %d designs", int(seed_total))

        for seed_i, gi in enumerate(init_seed, start=1):
            d = design_space.Design(dict(zip(space_spec.keys, GRID[gi])))
            lab, n_eps, mode = _label_via_cache_or_sim(gi, d, tester, epman, scenario_name)
            L.append(int(gi))
            y.append(int(lab))
            total_episodes_spent += int(n_eps)
            if seed_i == 1 or seed_i == seed_total or (seed_i % max(1, act.batch) == 0):
                logger.info(
                    "AL seeding progress | %d/%d | last_gi=%d label=%d mode=%s new_eps=%d total_eps=%d",
                    seed_i,
                    seed_total,
                    int(gi),
                    int(lab),
                    str(mode),
                    int(n_eps),
                    int(total_episodes_spent),
                )
    else:
        logger.info("Skipping AL init seeding; using %d cached labeled designs", len(L))
    
    logger.info("AL seed done | labeled=%d (safe=%d, unsafe=%d) | episodes=%d",
                len(y), int(np.sum(y)), int(len(y) - np.sum(y)), total_episodes_spent)


    # Trackers (from tracked version)
    trackers = dict(
        cycle=[], v_pred=[], d_jaccard=[], t_prov=[], t_delta=[],
        boundary_mass=[], sims_per_new_decision=[], mode_counts=[],
        labeled_total=[], decided_this_cycle=[], total_episodes_spent=[],
        q_used=[],
    )

    prev_mask = None
    t_last = None
    q = float(getattr(act, "t_prov_param", 0.8))
    eps_band = float(max(min(act.quantile_band_width, 0.5), 0.0))
    patience_ctr = 0

    # Iterations (cycles)
    for it in range(1, act.iters + 1):
        # Fit model
        model = _fit_model(GRID, L, y, seed=act.seed, cfg=act)

        # Scores (full + train)
        scores_all = model.predict_proba(GRID)[:, 1]
        scores_tr  = scores_all[train_idx]

        # Adaptive quantile update (estimate mass ≥ t_last)
        if act.adaptive_quantile and t_last is not None:
            f_hat = float(np.mean(scores_tr >= t_last))
            q = min(max(1.0 - f_hat, 0.01), 0.99)

        # Boundary selection
        t_prov = choose_t_prov(scores_tr, method=act.t_prov_method,
                               param=q, t_last=t_last)

        # Pre-label metrics
        mask_now = (scores_all >= t_prov)
        v_pred = float(np.mean(mask_now))
        d_j = float(_jaccard(prev_mask, mask_now))
        t_delta = float(abs(t_prov - t_last)) if t_last is not None else float("nan")
        boundary_mass = float(np.mean(np.abs(scores_all - t_prov) <= (eps_band if eps_band > 0 else 0.05)))

        # Candidate selection (risk-aware, diverse, near boundary)
        labeled_mask = np.zeros(len(GRID), dtype=bool)
        labeled_mask[L] = True
        cand01, _, _ = design_space.normalize_grid(GRID[train_idx])
        batch_rel = pick_batch_risk_aware(
            cand01, scores_tr, t_prov,
            already_labeled_mask=labeled_mask[train_idx],
            batch_size=min(act.batch, len(train_idx) - labeled_mask[train_idx].sum()),
            seed=act.seed + it,
            band_width=eps_band
        )

        # Stopping: no more candidates
        if len(batch_rel) == 0:
            logger.info("AL stop: no more candidates at cycle %d | labeled_total=%d | total_episodes=%d",
                        it, int(labeled_mask.sum()), total_episodes_spent)
            trackers["cycle"].append(int(it))
            trackers["v_pred"].append(v_pred)
            trackers["d_jaccard"].append(d_j)
            trackers["t_prov"].append(float(t_prov))
            trackers["t_delta"].append(t_delta)
            trackers["boundary_mass"].append(boundary_mass)
            trackers["sims_per_new_decision"].append(float("nan"))
            trackers["mode_counts"].append({})
            trackers["labeled_total"].append(int(len(L)))
            trackers["decided_this_cycle"].append(0)
            trackers["total_episodes_spent"].append(int(total_episodes_spent))
            trackers["q_used"].append(float(q))
            break

        new_global = list(train_idx[batch_rel])
        batch_total = len(new_global)
        logger.info("AL cycle %d | labeling %d designs", int(it), int(batch_total))

        # Label batch (cache-first), collect efficiency
        sim_eps_cycle = 0
        modes: Dict[str, int] = {}
        decided = 0
        for b_i, gi in enumerate(new_global, start=1):
            if gi in L:
                continue
            logger.info("AL cycle %d | design %d/%d | gidx=%d", int(it), int(b_i), int(batch_total), int(gi))
            d = design_space.Design(dict(zip(space_spec.keys, GRID[gi])))
            lab, n_eps, mode = _label_via_cache_or_sim(gi, d, tester, epman, scenario_name)
            L.append(int(gi))
            y.append(int(lab))
            sim_eps_cycle += int(n_eps)
            modes[mode] = modes.get(mode, 0) + 1
            decided += 1
            logger.info(
                "AL cycle %d | completed design %d/%d | gidx=%d | label=%d | mode=%s | new_eps=%d | labeled_total=%d",
                int(it),
                int(b_i),
                int(batch_total),
                int(gi),
                int(lab),
                str(mode),
                int(n_eps),
                int(len(L)),
            )

        total_episodes_spent += sim_eps_cycle
        sims_per_dec = sim_eps_cycle / max(decided, 1)

        # Log trackers this cycle
        trackers["cycle"].append(int(it))
        trackers["v_pred"].append(v_pred)
        trackers["d_jaccard"].append(d_j)
        trackers["t_prov"].append(float(t_prov))
        trackers["t_delta"].append(t_delta)
        trackers["boundary_mass"].append(boundary_mass)
        trackers["sims_per_new_decision"].append(float(sims_per_dec))
        trackers["mode_counts"].append(modes)
        trackers["labeled_total"].append(int(len(L)))
        trackers["decided_this_cycle"].append(int(decided))
        trackers["total_episodes_spent"].append(int(total_episodes_spent))
        trackers["q_used"].append(float(q))

        logger.info(
            "AL cycle %d / %d | q=%.3f t_prov=%.4f v_pred=%.3f dJ=%.4f boundary_mass=%.3f | decided=%d sims=%d (%.2f/dec) | labeled=%d",
            it, act.iters, q, t_prov, v_pred, d_j, boundary_mass, decided, sim_eps_cycle, sims_per_dec, int(labeled_mask.sum())
        )

        # Early stopping based on Jaccard stability
        if act.early_stop_patience > 0:
            if d_j <= act.early_stop_jaccard_eps:
                patience_ctr += 1
                if patience_ctr >= act.early_stop_patience:
                    logger.info(
                        "AL stop: early stability at cycle %d | dJ=%.6f <= eps=%.6f for %d/%d cycles",
                        int(it),
                        float(d_j),
                        float(act.early_stop_jaccard_eps),
                        int(patience_ctr),
                        int(act.early_stop_patience),
                    )
                    break
            else:
                patience_ctr = 0

        prev_mask = mask_now
        t_last = t_prov
        epman.flush()

    final_model = _fit_model(GRID, L, y, seed=act.seed, cfg=act)

    logger.info("AL done | final labeled=%d (safe=%d, unsafe=%d) | total_episodes=%d",
                len(L), int(np.sum(y)), int(len(y) - np.sum(y)), total_episodes_spent)

    return dict(
        model=final_model,
        labeled_idx=np.array(L, dtype=int),
        labels=np.array(y, dtype=int),
        train_idx=train_idx,
        trackers=trackers,
        algo_version=algo_version,
    )
