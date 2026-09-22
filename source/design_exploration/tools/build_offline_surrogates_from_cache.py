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
"""Label cached designs, split train/calibration/test data, and train five families.

RF, NN ensemble, Gaussian GP, Student-t GP and QRF run in subprocesses.
Each invocation creates a fresh dataset unless --split-path reuses saved labels
and splits. Reused datasets must match the requested scenario and label policy.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

# Support both the documented module command and direct script execution.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from source.design_exploration.commons import design_space
from source.design_exploration.commons.episodes.episode_manager import EpisodeManager
from source.design_exploration.commons.episodes.episode_result import EpisodeResult
from source.design_exploration.commons.offline_labeling import (
    make_min_dist_violation_function,
    offline_sprt_label,
)
from source.design_exploration.commons.python_paths import (
    get_python311_path,
    require_python311_path,
)
from source.design_exploration.commons.scenario_defaults import SCENARIO_2
from source.design_exploration.commons.scenario_labeling import SPRTConfig, TestResult
from source.design_exploration.commons.surrogate_storage import (
    default_surrogate_root,
    prepare_run_dir,
)

logger = logging.getLogger(__name__)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario-name", default=str(SCENARIO_2["name"]))
    ap.add_argument("--space-spec-name", default=design_space.SPACE_SPEC.name)
    ap.add_argument("--episode-root", default=str(_REPO_ROOT / "episode_dataset"))
    ap.add_argument("--surrogate-root", default=str(default_surrogate_root(Path(__file__))))
    ap.add_argument("--algo-version", default="builder_v1")
    ap.add_argument("--python311", help="QRF interpreter (otherwise PYTHON311 or the current interpreter if QRF is installed).")
    ap.add_argument("--max-train-workers", type=int, default=8, help="Maximum concurrent trainer subprocesses; use 1 for sequential training.")
    ap.add_argument("--calib-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--d-safe", type=float, default=0.5)
    ap.add_argument("--split-path", type=Path, help="Reuse an existing split and its labels without modifying either; default: create a fresh dataset.")
    ap.add_argument("--sprt-P", dest="sprt_p", type=float, default=0.05)
    ap.add_argument("--sprt-C", dest="sprt_c", type=float, default=0.95)
    ap.add_argument("--sprt-p1-factor", type=float, default=0.5)
    ap.add_argument("--sprt-beta-err", type=float, default=0.2)
    ap.add_argument("--sprt-max-iters", type=int, default=80)
    ap.add_argument("--sprt-batch-size", type=int, default=10)
    ap.add_argument("--skip-qrf", action="store_true", help="Train only the four standard families.")
    args = ap.parse_args(argv)
    if args.max_train_workers < 1:
        ap.error("--max-train-workers must be positive")
    if not (0 <= args.calib_frac <= 1 and 0 <= args.test_frac <= 1
            and args.calib_frac + args.test_frac < 1):
        ap.error("Split fractions must be non-negative and sum to less than 1")
    if not np.isfinite(args.d_safe):
        ap.error("--d-safe must be finite")
    for name in ("sprt_p", "sprt_c", "sprt_p1_factor", "sprt_beta_err"):
        if not 0 < getattr(args, name) < 1:
            ap.error(f"{name} must be between 0 and 1")
    if args.sprt_max_iters < 1 or args.sprt_batch_size < 1:
        ap.error("SPRT iteration and batch limits must be positive")
    return args


def _trainer_jobs(args):
    jobs = [(name, sys.executable) for name in ("rf", "nn", "hetgp", "hetgpt")]
    if not args.skip_qrf:
        py311 = get_python311_path(args.python311)
        if py311 or importlib.util.find_spec("quantile_forest") is None:
            executable = require_python311_path(
                py311, cli_flag="--python311", purpose="QRF offline training subprocess"
            )
        else:
            executable = sys.executable
        jobs.append(("qrf", executable))
    return jobs


def _run_trainer_subprocess(name, split_path, args, python_executable):
    script_path = Path(__file__).with_name("train_surrogate_subprocess.py")
    cmd = [
        python_executable, str(script_path), "--trainer", name,
        "--split-path", str(split_path), "--episode-root", args.episode_root,
        "--surrogate-root", args.surrogate_root, "--algo-version", args.algo_version,
        "--seed", str(args.seed), "--d-safe", str(args.d_safe),
    ]
    logger.info("[offline_surr][%s] starting trainer", name)
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{name} trainer failed (exit {exc.returncode}):\n{exc.stderr or exc.stdout}"
        ) from exc
    try:
        artifacts = json.loads(proc.stdout)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"{name} trainer returned invalid artifact JSON: {proc.stdout!r}") from exc
    if not isinstance(artifacts, dict) or not artifacts.get("model"):
        raise RuntimeError(f"{name} trainer returned no model artifact")
    logger.info("[offline_surr][%s] completed | model=%s", name, artifacts["model"])
    return artifacts


def _label_train_indices(epman, train_idx, sprt_cfg, d_safe):
    labeled = []
    violation_fn = make_min_dist_violation_function(d_safe)
    for gi in train_idx:
        cached = epman.episodes(int(gi), consume=False)
        if not cached:
            continue
        episodes = []
        for record in cached:
            ep = EpisodeResult()
            payload = record.get("payload", {})
            ep.result = payload if isinstance(payload, dict) else {"payload": payload}
            episodes.append(ep)
        if episodes:
            outcome = offline_sprt_label(episodes, sprt_cfg, violation_fn)
            if outcome.label is not None:
                labeled.append((int(gi), int(outcome.label)))
    return labeled


def _split_labeled(
    labeled_pairs: Sequence[Tuple[int, int]], calib_frac: float, test_frac: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    if calib_frac < 0.0 or test_frac < 0.0 or (calib_frac + test_frac) > 1.0:
        raise ValueError(
            f"Expected non-negative split fractions with calib_frac + test_frac <= 1.0, "
            f"got calib_frac={calib_frac}, test_frac={test_frac}"
        )

    arr = np.asarray(labeled_pairs, dtype=int)
    if arr.size == 0:
        return [], [], []

    rng = np.random.RandomState(seed)

    def _allocate_counts(n_items: int) -> Tuple[int, int, int]:
        desired = np.asarray(
            [
                float(calib_frac) * n_items,
                float(test_frac) * n_items,
                max(0.0, 1.0 - float(calib_frac) - float(test_frac)) * n_items,
            ],
            dtype=float,
        )
        counts = np.floor(desired).astype(int)
        remainder = int(n_items - counts.sum())
        if remainder > 0:
            frac = desired - counts
            order = np.argsort(-frac, kind="mergesort")
            for idx in order[:remainder]:
                counts[int(idx)] += 1
        return int(counts[0]), int(counts[1]), int(counts[2])

    train_parts: List[np.ndarray] = []
    calib_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []

    for label in sorted(int(v) for v in np.unique(arr[:, 1])):
        rows = arr[arr[:, 1] == label].copy()
        rng.shuffle(rows)
        n_calib, n_test, _ = _allocate_counts(len(rows))
        calib_parts.append(rows[:n_calib])
        test_parts.append(rows[n_calib : n_calib + n_test])
        train_parts.append(rows[n_calib + n_test :])

    def _merge_parts(parts: List[np.ndarray]) -> np.ndarray:
        non_empty = [part for part in parts if part.size]
        if not non_empty:
            return np.empty((0, 2), dtype=int)
        merged = np.concatenate(non_empty, axis=0).astype(int, copy=False)
        rng.shuffle(merged)
        return merged

    train = _merge_parts(train_parts)
    calib = _merge_parts(calib_parts)
    test = _merge_parts(test_parts)
    return (
        train[:, 0].tolist(),
        calib[:, 0].tolist(),
        test[:, 0].tolist(),
    )


def _label_policy(args):
    return {
        "P": args.sprt_p, "C": args.sprt_c, "p1_factor": args.sprt_p1_factor,
        "beta_err": args.sprt_beta_err, "global_max_iters": args.sprt_max_iters,
        "batch_size": args.sprt_batch_size,
    }


def _load_dataset(split_path, args, space_spec):
    with split_path.open(encoding="utf-8") as handle:
        split = json.load(handle)
    labels_path = Path(split["labels_path"]).expanduser().resolve()
    with labels_path.open(encoding="utf-8") as handle:
        labels = json.load(handle)
    for name, manifest in (("split", split), ("labels", labels)):
        for key, expected in (("scenario_name", args.scenario_name),
                              ("space_spec_name", space_spec.name)):
            if manifest.get(key) != expected:
                raise ValueError(f"Reused {name} {key} does not match {expected!r}")
    if labels.get("d_safe") != args.d_safe or labels.get("sprt") != _label_policy(args):
        raise ValueError("Reused labels do not match the requested d_safe/SPRT policy")
    labeled = [(int(gi), int(label)) for gi, label in labels["labels"].items()]
    valid_labels = {int(TestResult.SAFE), int(TestResult.UNSAFE)}
    if not labeled or any(label not in valid_labels for _, label in labeled):
        raise ValueError("Reused labels must contain SAFE/UNSAFE decisions")
    splits = [split[key] for key in ("train", "calib", "test")]
    ids = [gi for part in splits for gi in part]
    grid_size = len(space_spec.build_grid())
    if any(type(gi) is not int or not 0 <= gi < grid_size for gi in ids):
        raise ValueError("Split contains invalid design indices")
    if len(ids) != len(set(ids)) or set(ids) != {gi for gi, _ in labeled}:
        raise ValueError("Splits must be disjoint and cover each labeled design exactly once")
    if not splits[0]:
        raise ValueError("Training split must not be empty")
    return splits, labeled, labels_path


def _write_dataset(run_dir, args, space_spec, splits, labeled):
    dataset_id = run_dir.name
    labels_path = run_dir / "offline_labels.json"
    split_path = run_dir / "offline_splits.json"
    common = {
        "dataset_id": dataset_id, "dataset_run_id": dataset_id,
        "scenario_name": args.scenario_name, "space_spec_name": space_spec.name,
    }
    split = dict(common, split_id=dataset_id, space_spec_keys=list(space_spec.keys),
                 grid_size=len(space_spec.build_grid()), labels_path=str(labels_path),
                 **dict(zip(("train", "calib", "test"), splits)))
    labels = dict(common, labels_id=dataset_id,
                  episode_root=str(Path(args.episode_root) / args.scenario_name),
                  algo_version=args.algo_version, d_safe=args.d_safe,
                  sprt=_label_policy(args), labels={str(gi): label for gi, label in labeled})
    split_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    labels_path.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return split_path, labels_path


def build_offline_surrogates(argv=None):
    args = _parse_args(argv)
    jobs = _trainer_jobs(args)
    space_spec = design_space.get_space_spec(args.space_spec_name)
    if args.split_path is not None:
        split_path = args.split_path.expanduser().resolve()
        splits, labeled, labels_path = _load_dataset(split_path, args, space_spec)
        dataset_run_dir = split_path.parent
    else:
        grid = space_spec.build_grid()
        epman = EpisodeManager(
            root_dir=str(Path(args.episode_root) / args.scenario_name),
            scenario_name=args.scenario_name, space_spec=space_spec, GRID=grid,
            flush_every=50,
        )
        sprt_cfg = SPRTConfig(**_label_policy(args))
        logger.info("[offline_surr] labeling cached designs | scenario=%s", args.scenario_name)
        labeled = _label_train_indices(epman, range(len(grid)), sprt_cfg, args.d_safe)
        if not labeled:
            raise RuntimeError("No labeled designs found in cache; cannot train offline surrogates.")
        splits = _split_labeled(labeled, args.calib_frac, args.test_frac, args.seed)
        if not splits[0]:
            raise ValueError("Training split must not be empty")
        dataset_run_dir = Path(prepare_run_dir(args.surrogate_root, "offline_builder")).resolve()
        split_path, labels_path = _write_dataset(dataset_run_dir, args, space_spec, splits, labeled)

    labels_by_id = dict(labeled)
    for name, ids in zip(("train", "calib", "test"), splits):
        unsafe = sum(labels_by_id[gi] == int(TestResult.UNSAFE) for gi in ids)
        logger.info("[offline_surr] %s=%d | safe=%d unsafe=%d", name, len(ids), len(ids) - unsafe, unsafe)

    with ThreadPoolExecutor(max_workers=min(args.max_train_workers, len(jobs))) as executor:
        futures = [executor.submit(_run_trainer_subprocess, name, split_path, args, executable)
                   for name, executable in jobs]
        for future in as_completed(futures):
            future.result()

    artifacts = {
        "artifact_bundle_id": dataset_run_dir.name,
        "dataset_id": dataset_run_dir.name,
        "dataset_run_id": dataset_run_dir.name,
        "split_id": dataset_run_dir.name,
        "labels_id": dataset_run_dir.name,
        "split_path": str(split_path),
        "labels_path": str(labels_path),
        "dataset_run_dir": str(dataset_run_dir),
        "scenario_name": args.scenario_name,
        "space_spec_name": space_spec.name,
    }
    artifacts_path = dataset_run_dir / "offline_artifacts.json"
    artifacts_path.write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")
    logger.info("[offline_surr] completed | dataset manifest=%s", artifacts_path)
    return artifacts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_offline_surrogates()
