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
Run one offline surrogate trainer in a subprocess and serialize artifact paths.

This script is used by build_offline_surrogates_from_cache.py when offline
training is parallelized across subprocesses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Sequence

# Ensure the repository root (folder containing "source") is on sys.path.
_HERE = Path(__file__).resolve()
for cand in [_HERE] + list(_HERE.parents):
    if (cand / "source").is_dir():
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
        break

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run one offline surrogate trainer in a subprocess.")
    ap.add_argument("--trainer", required=True, help="Trainer key: rf, nn, hetgp, hetgpt, qrf.")
    ap.add_argument("--split-path", required=True, help="Path to offline_splits.json.")
    ap.add_argument("--episode-root", required=True, help="Episode dataset root, e.g. repo/episode_dataset.")
    ap.add_argument("--surrogate-root", dest="surrogate_root", required=False, help="Root directory where surrogate run folders are written.")
    ap.add_argument("--learner-root", dest="surrogate_root", required=False, help=argparse.SUPPRESS)
    ap.add_argument("--algo-version", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--d-safe", required=True, type=float)
    args = ap.parse_args()
    if not args.surrogate_root:
        raise SystemExit("--surrogate-root is required")
    return args


def _load_split(path: str) -> tuple[list[int], list[int], list[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("train", [])), list(data.get("calib", [])), list(data.get("test", []))


def _resolve_trainer(name: str) -> tuple[Callable[[Sequence[int], Sequence[int], Sequence[int]], Dict[str, str]], object]:
    if name == "rf":
        from source.design_exploration.surrogates.rf_cls import offline_train_rf as mod

        fn_name = "train_rf_offline"
    elif name == "nn":
        from source.design_exploration.surrogates.nn_cls import offline_train_nn_ensemble as mod

        fn_name = "train_nn_ensemble_offline"
    elif name == "hetgp":
        from source.design_exploration.surrogates.hetGP import offline_train_gp as mod

        fn_name = "train_gp_offline"
    elif name == "hetgpt":
        from source.design_exploration.surrogates.hetGPt import offline_train_student_t_gp as mod

        fn_name = "train_student_t_gp_offline"
    elif name == "qrf":
        from source.design_exploration.surrogates.qrf import train_qrf_offline as mod

        fn_name = "train_qrf_offline"
    else:
        raise ValueError(f"Unknown trainer {name!r}. Expected one of: {sorted(['rf', 'nn', 'hetgp', 'hetgpt', 'qrf'])}")

    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise RuntimeError(f"Trainer function {fn_name!r} not found for trainer {name!r}.")
    return fn, mod


def main() -> None:
    args = _parse_args()
    train_idx, calib_idx, test_idx = _load_split(args.split_path)
    trainer_fn, _trainer_mod = _resolve_trainer(args.trainer)
    artifacts = trainer_fn(
        train_idx,
        calib_idx,
        test_idx,
        episode_root=args.episode_root,
        surrogate_root=args.surrogate_root,
        algo_version=args.algo_version,
        split_path=args.split_path,
        seed=args.seed,
        d_safe=args.d_safe,
    )
    sys.stdout.write(json.dumps(artifacts))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
