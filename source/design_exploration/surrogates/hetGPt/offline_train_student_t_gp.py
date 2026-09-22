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
Offline training for the heteroscedastic Student-t GP surrogate.

Fits the surrogate on a train split of cached episodes, calibrates bias/variance
scale on a calibration split, and persists the artifacts.
"""
from __future__ import annotations

import json
from dataclasses import asdict
import logging
import os
from pathlib import Path
from typing import Dict, Sequence

import joblib
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from scipy.stats import t as stats_t
from source.design_exploration.surrogates.hetGPt.heteroscedastic_student_t_gp import (
    _t_scale_from_variance,
)

from source.design_exploration.commons import design_space
from source.design_exploration.commons.episodes.episode_manager import EpisodeManager
from source.design_exploration.surrogates.margin_calibration import (
    MarginCalibrationCfg,
    compute_student_t_calibration,
)
from source.design_exploration.surrogates.hetGPt.heteroscedastic_student_t_gp import (
    HeteroStudentTCfg,
    build_margin_dataset,
    predict_violation_probability,
    train_from_episode_cache,
)
from source.design_exploration.commons.surrogate_storage import (
    default_surrogate_root,
    latest_run_dir,
    prepare_run_dir,
)
from source.design_exploration.commons.traceability import split_trace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
_REPO_ROOT = next((cand for cand in [_HERE] + list(_HERE.parents) if (cand / "source").is_dir()), _HERE.parent)

ALGO_VERSION = "approach1_v1"
EPISODE_ROOT = str(_REPO_ROOT / "episode_dataset")
SURROGATE_ROOT = str(default_surrogate_root(_HERE))
OFFLINE_DATASET_NAME = "offline_builder"
MODEL_OUT = "student_t_gp_offline.joblib"
TRAINER_VERSION = "v1"
D_SAFE = 0.5
SEED = 13
MIN_EPS_PER_DESIGN = 30


def _default_dataset_run_dir(surrogate_root: str | None = None) -> Path:
    root = surrogate_root or SURROGATE_ROOT
    run_dir = latest_run_dir(root, OFFLINE_DATASET_NAME)
    if run_dir is None:
        raise FileNotFoundError(f"No offline dataset run found under {Path(root) / OFFLINE_DATASET_NAME}")
    return run_dir


def _load_split_manifest(split_path: str | None = None, surrogate_root: str | None = None) -> Dict[str, object]:
    path = Path(split_path) if split_path else (_default_dataset_run_dir(surrogate_root) / "offline_splits.json")
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_space_spec(split_manifest: Dict[str, object], spec_name: str | None = None) -> design_space.SpaceSpec:
    resolved_name = split_manifest.get("space_spec_name") or spec_name
    if not resolved_name:
        raise KeyError("space_spec_name missing from split manifest and not provided explicitly")
    return design_space.get_space_spec(str(resolved_name))


def train_student_t_gp_offline(
    train_idx: Sequence[int],
    calib_idx: Sequence[int],
    test_idx: Sequence[int] | None = None,
    *,
    space_spec_name: str | None = None,
    episode_root: str | None = None,
    surrogate_root: str | None = None,
    algo_version: str | None = None,
    split_path: str | None = None,
    seed: int | None = None,
    d_safe: float | None = None,
    min_episodes_per_design: int | None = None,
) -> Dict[str, str]:
    episode_root = episode_root or EPISODE_ROOT
    surrogate_root = surrogate_root or SURROGATE_ROOT
    algo_version = algo_version or ALGO_VERSION
    seed = SEED if seed is None else int(seed)
    d_safe = D_SAFE if d_safe is None else float(d_safe)
    min_episodes_per_design = MIN_EPS_PER_DESIGN if min_episodes_per_design is None else int(min_episodes_per_design)
    split_manifest = _load_split_manifest(split_path, surrogate_root)
    scenario_name = split_manifest.get("scenario_name")
    if not scenario_name:
        raise KeyError("scenario_name missing from split manifest")
    space_spec = _resolve_space_spec(split_manifest, space_spec_name)
    grid = space_spec.build_grid()
    ep_root = os.path.join(episode_root, scenario_name)
    gp_cfg = HeteroStudentTCfg(seed=seed, d_safe=d_safe)
    calibration_cfg = MarginCalibrationCfg()
    epman = EpisodeManager(
        root_dir=ep_root,
        scenario_name=str(scenario_name),
        space_spec=space_spec,
        GRID=grid,
        flush_every=50,
    )

    mean_model, var_model, dataset_train = train_from_episode_cache(
        epman=epman,
        cfg=gp_cfg,
        min_episodes_per_design=min_episodes_per_design,
        allowed_idx=train_idx,
    )

    allowed_union = np.unique(
        np.asarray(list(train_idx) + list(calib_idx) + ([] if test_idx is None else list(test_idx)), dtype=int)
    )
    dataset_all = build_margin_dataset(
        epman,
        cfg=gp_cfg,
        min_episodes_per_design=min_episodes_per_design,
        norm_grid=grid,
        allowed_idx=allowed_union,
    )

    mu_raw, std_epistemic_raw, var_aleatoric_raw, _ = predict_violation_probability(
        mean_model, var_model, grid, norm_params=(dataset_train.norm_lo, dataset_train.norm_span), df=gp_cfg.df
    )
    var_total_raw = np.square(std_epistemic_raw) + var_aleatoric_raw
    calib_idx = np.asarray(calib_idx, dtype=int)
    test_idx_arr = np.asarray(test_idx, dtype=int) if test_idx is not None else np.array([], dtype=int)
    calib_eps_total = int(sum(len(dataset_all.margins[int(gi)]) for gi in calib_idx if int(gi) in dataset_all.margins))

    if len(calib_idx) >= 10 and calib_eps_total >= 200:
        bias, var_scale, calib_used_designs, calib_used_eps, cov_lo, cov_mid, cov_hi, calib_used_idx = compute_student_t_calibration(
            mu_raw,
            var_total_raw,
            dataset_all.margins,
            calib_idx,
            gp_cfg.df,
            calibration_cfg.calib_gamma,
            calibration_cfg.calib_max_iters,
            calibration_cfg.calib_tol,
            calibration_cfg.calib_var_scale_bounds,
        )
    else:
        logger.warning("[student_t_gp_offline][calib] insufficient calibration data (designs=%d eps=%d); using bias=0 var_scale=1", len(calib_idx), calib_eps_total)
        bias, var_scale = 0.0, 1.0
        calib_used_designs, calib_used_eps, cov_lo, cov_mid, cov_hi, calib_used_idx = 0, calib_eps_total, float("nan"), float("nan"), float("nan"), np.array([], dtype=int)

    # Optional log-loss on calib/test set
    def _compute_logloss(idx_arr: np.ndarray, name: str):
        labels = []
        preds = []
        used_designs = 0
        for gi in idx_arr:
            if int(gi) not in dataset_all.margins:
                continue
            mvals = dataset_all.margins[int(gi)]
            if mvals is None or len(mvals) == 0:
                continue
            idx = int(gi)
            if idx >= len(mu_raw):
                continue
            mu_adj = mu_raw[idx] + bias
            var_adj = var_total_raw[idx] * var_scale
            scale = _t_scale_from_variance(var_adj, gp_cfg.df)
            p_unsafe = float(stats_t.cdf(0.0, df=gp_cfg.df, loc=mu_adj, scale=scale))
            y_eps = (np.asarray(mvals, dtype=float) < 0.0).astype(int)
            if y_eps.size == 0:
                continue
            p_use = float(np.clip(p_unsafe, 1e-6, 1 - 1e-6))
            labels.extend(y_eps.tolist())
            preds.extend([p_use] * int(y_eps.size))
            used_designs += 1
        if labels and preds:
            labels_arr = np.asarray(labels)
            preds_arr = np.asarray(preds)
            uniq = np.unique(labels_arr)
            val = float(log_loss(labels_arr, preds_arr, labels=[0, 1]))
            acc = float(accuracy_score(labels_arr, (preds_arr >= 0.5).astype(int)))
            auc = None
            if len(uniq) >= 2:
                try:
                    auc = float(roc_auc_score(labels_arr, preds_arr))
                except Exception:
                    auc = None
            else:
                logger.info(
                    "[student_t_gp_offline] WARNING: %s set is single-class labels=%s; AUC=N/A",
                    name.upper(),
                    uniq.tolist(),
                )
            logger.info(
                "[student_t_gp_offline] %s logloss=%.4f acc=%.4f auc=%s on %d designs (%d episodes)",
                name,
                val,
                acc,
                f"{auc:.4f}" if auc is not None else "n/a",
                used_designs,
                len(labels_arr),
            )
            return val, acc, auc
        return None, None, None

    calib_logloss, calib_acc, calib_auc = _compute_logloss(calib_idx, "calib")
    test_logloss, test_acc, test_auc = _compute_logloss(test_idx_arr, "test") if test_idx_arr.size else (None, None, None)

    artifacts = {
        "mean_model": mean_model,
        "var_model": var_model,
        "norm_lo": dataset_train.norm_lo,
        "norm_span": dataset_train.norm_span,
        "bias": bias,
        "var_scale": var_scale,
        "df": gp_cfg.df,
        "test_idx": [int(x) for x in test_idx_arr],
        "calib_used_designs": calib_used_designs,
        "calib_used_eps": calib_used_eps,
        "calib_used_idx": calib_used_idx,
        "coverage_bracket": {"lo": cov_lo, "mid": cov_mid, "hi": cov_hi},
        "cfg": {"gp_cfg": gp_cfg, "calibration_cfg": asdict(calibration_cfg)},
        "calib_logloss": calib_logloss,
        "test_logloss": test_logloss,
        "calib_accuracy": calib_acc,
        "calib_auc": calib_auc,
        "test_accuracy": test_acc,
        "test_auc": test_auc,
    }
    out_dir = prepare_run_dir(surrogate_root, "hetgpt")
    model_path = os.path.join(out_dir, MODEL_OUT)
    joblib.dump(artifacts, model_path)

    meta = {
        "scenario": scenario_name,
        "space_spec_name": space_spec.name,
        "space_spec_keys": list(space_spec.keys),
        "episode_root": ep_root,
        "surrogate_root": surrogate_root,
        "run_dir": out_dir,
        "dataset_manifest_path": str(Path(split_path).with_name("offline_artifacts.json")) if split_path else None,
        "split_path": split_path,
        "trainer_version": TRAINER_VERSION,
        "algo_version": algo_version,
        "model_id": Path(out_dir).name,
        "model_run_id": Path(out_dir).name,
        "model_family": Path(out_dir).parent.name,
        "min_eps_per_design": int(min_episodes_per_design),
        "train_idx": [int(x) for x in train_idx],
        "calib_idx": [int(x) for x in calib_idx],
        "test_idx": [int(x) for x in test_idx_arr],
        "calib_eps_total": calib_eps_total,
        "model_path": model_path,
        "bias": bias,
        "var_scale": var_scale,
        "df": gp_cfg.df,
        "calib_logloss": calib_logloss,
        "test_logloss": test_logloss,
        "calib_accuracy": calib_acc,
        "calib_auc": calib_auc,
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "trace": {
            "model_id": Path(out_dir).name,
            "model_run_id": Path(out_dir).name,
            "model_family": Path(out_dir).parent.name,
            "dataset_manifest_path": str(Path(split_path).with_name("offline_artifacts.json")) if split_path else None,
            "split": split_trace(split_path) if split_path else None,
        },
    }
    meta_path = os.path.join(out_dir, os.path.splitext(MODEL_OUT)[0] + "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("[student_t_gp_offline] saved model=%s meta=%s", model_path, meta_path)
    return {
        "model": model_path,
        "meta": meta_path,
        "bias": bias,
        "var_scale": var_scale,
        "df": gp_cfg.df,
    }
