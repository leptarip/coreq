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
Convenience launcher for Random Forest (RF) active learning on the design grid.

Adjust the constants below (scenario, config path, storage roots) or override
via CLI flags when orchestrated by start_al_script.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import argparse
from pathlib import Path

import numpy as np

import source.simulation_environment.cfg.cfg_parser as cfg_parser
from source.design_exploration.commons.design_space import SPACE_SPEC
from source.design_exploration.commons.offline_labeling import make_min_dist_violation_function
from source.design_exploration.commons.scenario_defaults import SCENARIO_1, SCENARIO_2
from source.design_exploration.commons.surrogate_storage import default_surrogate_root
from source.design_exploration.surrogates.rf_cls.rf_active_learning import (
    ActiveCfg,
    active_grid_learning,
)
from source.design_exploration.commons.scenario_labeling import (
    ScenarioTester,
    SPRTConfig,
)
from source.design_exploration.commons.persistence import (
    persist_active_learning_results,
)
from source.design_exploration.simulator_interface.pooled_sim_interface import PooledSimInterface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
_HERE = Path(__file__).resolve()
_REPO_ROOT = next((cand for cand in [_HERE] + list(_HERE.parents) if (cand / "source").is_dir()), _HERE.parent)

# === User-adjustable defaults ===
SCENARIO_NAME = SCENARIO_2["name"]
SCENARIO_CFG_PATH = SCENARIO_2["config"]

ALGO_VERSION = "v1"
SEED = 13
D_SAFE = 0.5

OUT_ROOT = str(default_surrogate_root(_HERE))
LEARNER_DIR = os.path.join(OUT_ROOT, "al_rf")
SIM_OUT_DIR = os.path.join(LEARNER_DIR, "carla_sim_out", SCENARIO_NAME)
EPISODE_CACHE_ROOTS = [
    str(_REPO_ROOT / "episode_dataset"),
]
NUM_WORKERS = 3
CARLA_PATH = os.environ.get("CARLA_PATH", "")

ACT_CFG = ActiveCfg(
    seed=SEED,
    init_k=32,
    batch=16,
    iters=40,
    flush_every=100,
    calibrate=True,
    calib_method="sigmoid",
    calib_cv=5,
)


def _apply_cli_overrides():
    """
    Allow scenario config/name overrides passed via CLI for orchestration scripts.
    """
    parser = argparse.ArgumentParser(description='Active learning for the RF classifier on one scenario.')
    parser.add_argument("--scenario-config", dest="scenario_cfg_path")
    parser.add_argument("--scenario-name", dest="scenario_name")
    parser.add_argument("--num-workers", dest="num_workers", type=int)
    parser.add_argument("--episode-cache-root", dest="episode_cache_roots", action="append")
    parser.add_argument("--carla-path", dest="carla_path", help="CARLA server executable (else $CARLA_PATH).")
    parser.add_argument("--out-root", dest="out_root", help="Output root (default: output_surrogate_models/).")
    args = parser.parse_args()

    global SCENARIO_CFG_PATH, SCENARIO_NAME, SIM_OUT_DIR, NUM_WORKERS, EPISODE_CACHE_ROOTS, CARLA_PATH, OUT_ROOT, LEARNER_DIR

    if args.scenario_cfg_path:
        SCENARIO_CFG_PATH = args.scenario_cfg_path
        inferred = Path(SCENARIO_CFG_PATH).stem
        if inferred.startswith("config_"):
            inferred = inferred[len("config_"):]
        inferred = inferred.replace("_debug", "")
        if not args.scenario_name:
            SCENARIO_NAME = inferred

    if args.scenario_name:
        SCENARIO_NAME = args.scenario_name
        if not args.scenario_cfg_path:
            SCENARIO_CFG_PATH = {SCENARIO_1["name"]: SCENARIO_1["config"],
                                 SCENARIO_2["name"]: SCENARIO_2["config"]}[SCENARIO_NAME]

    if args.num_workers is not None:
        NUM_WORKERS = max(1, int(args.num_workers))

    if args.carla_path:
        CARLA_PATH = args.carla_path

    if args.episode_cache_roots:
        EPISODE_CACHE_ROOTS = [str(Path(p).expanduser()) for p in args.episode_cache_roots if p]

    if args.out_root:
        OUT_ROOT = str(Path(args.out_root).expanduser())
        LEARNER_DIR = os.path.join(OUT_ROOT, os.path.basename(LEARNER_DIR))

    SIM_OUT_DIR = os.path.join(LEARNER_DIR, "carla_sim_out", SCENARIO_NAME)


def run_rf_active_learning():
    _apply_cli_overrides()

    # Ensure repository root on sys.path when run directly
    for cand in [_HERE] + list(_HERE.parents):
        if (cand / "source").is_dir() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
            break

    scenario_cfg = cfg_parser.parse(SCENARIO_CFG_PATH)
    space_spec = SPACE_SPEC
    GRID = space_spec.build_grid()
    train_idx = np.arange(len(GRID), dtype=int)
    logger.info("[grid] using full GRID for active learning | n_designs=%d", len(train_idx))
    logger.info(
        "[rf_al] launch config | scenario=%s | seed=%d | workers=%d | init_k=%d | iters=%d | batch=%d | calibrate=%s",
        SCENARIO_NAME,
        SEED,
        NUM_WORKERS,
        ACT_CFG.init_k,
        ACT_CFG.iters,
        ACT_CFG.batch,
        str(ACT_CFG.calibrate),
    )
    logger.info("[rf_al] cache roots | %s", json.dumps(EPISODE_CACHE_ROOTS, indent=2))
    logger.info("[rf_al] output root | %s", LEARNER_DIR)

    sim_iface = PooledSimInterface(
        output_folder=SIM_OUT_DIR,
        carla_path=CARLA_PATH,
        algo_version=ALGO_VERSION,
        num_workers=NUM_WORKERS,
        seed_base=SEED,
    )
    try:
        tester = ScenarioTester(
            sim_iface,
            SCENARIO_NAME,
            scenario_cfg,
            SPRTConfig(P=0.05),
            make_min_dist_violation_function(D_SAFE),
        )

        result = active_grid_learning(
            learner_dir=LEARNER_DIR,
            scenario_name=SCENARIO_NAME,
            tester=tester,
            act=ACT_CFG,
            GRID=GRID,
            train_idx=train_idx,
            space_spec=space_spec,
            algo_version=ALGO_VERSION,
            episode_cache_roots=EPISODE_CACHE_ROOTS,
        )

        paths = persist_active_learning_results(
            result,
            out_dir=os.path.join(LEARNER_DIR, SCENARIO_NAME),
            run_id=None,
            meta={
                "scenario": SCENARIO_NAME,
                "model_family": "rf",
                "space_spec_name": space_spec.name,
                "space_spec_keys": list(space_spec.keys),
                "seed": SEED,
                "d_safe": D_SAFE,
                "n_train": int(len(train_idx)),
                "sampling_pool": "full_grid",
                "episode_cache_roots": list(EPISODE_CACHE_ROOTS),
            },
        )

        logger.info("RF active learning completed.")
        logger.info("Artifacts: %s", json.dumps(paths, indent=2))
    finally:
        try:
            sim_iface.close()
        except Exception:
            pass


if __name__ == "__main__":
    run_rf_active_learning()
