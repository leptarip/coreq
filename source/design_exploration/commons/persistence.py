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
# persistence.py
from __future__ import annotations
import os, io, json, time, hashlib, tempfile
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _atomic_write_bytes(path: str, data: bytes) -> None:
    _ensure_dir(os.path.dirname(path))
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def _atomic_write_text(path: str, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_write_json(path: str, obj: Any) -> None:
    def _default(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        try:
            return dict(o)
        except Exception:
            return str(o)
    _atomic_write_text(path, json.dumps(obj, default=_default, indent=2, sort_keys=True))

def _json_default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    try:
        return dict(o)
    except Exception:
        return str(o)

def _hash_small(obj: Any) -> str:
    try:
        js = json.dumps(obj, sort_keys=True, default=_json_default).encode("utf-8")
    except Exception:
        js = str(obj).encode("utf-8")
    return hashlib.sha1(js).hexdigest()[:10]


def persist_active_learning_results(
    result: Dict[str, Any],
    out_dir: str,
    *,
    run_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    model_filename: Optional[str] = None,
    meta_filename: Optional[str] = None,
) -> Dict[str, str]:
    """
    Persist the output dict returned by active_grid_learning(...) to disk.

    Parameters
    ----------
    result : dict
        Expected keys:
          - 'model' (sklearn estimator, maybe CalibratedClassifierCV)
          - 'labeled_idx' (np.ndarray[int])
          - 'labels' (np.ndarray[int])
          - 'train_idx' (np.ndarray[int])
          - 'trackers' (dict of lists / JSON-serializable)
          - 'algo_version' (str)
    out_dir : str
        Root directory where a run subfolder will be created.
    run_id : str, optional
        Folder name for this run. If None, a timestamped id is used.
    meta : dict, optional
        Extra metadata to save (e.g., scenario_name, seed, designspec hash, cfg).

    Returns
    -------
    paths : dict
        Mapping of artifact names to file paths.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    # Try to make a readable, unique run id
    rid = run_id or f"al_run_{ts}_{_hash_small({k: type(result.get(k)).__name__ for k in result.keys()})}"
    run_dir = os.path.join(out_dir, rid)
    _ensure_dir(run_dir)

    paths: Dict[str, str] = {}
    meta_in = dict(meta or {})
    model_family = str(meta_in.get("model_family") or "").strip().lower() or None
    model_file = model_filename or (f"{model_family}_active.joblib" if model_family else "model.joblib")

    # ---- Save model ----
    model = result.get("model", None)
    if model is not None and joblib is not None:
        p_model = os.path.join(run_dir, model_file)
        bio = io.BytesIO()
        joblib.dump(model, bio)  # write to memory first (atomic on replace)
        _atomic_write_bytes(p_model, bio.getvalue())
        paths["model"] = p_model

        # also store model class + params for quick inspection
        try:
            model_info = {
                "class": model.__class__.__name__,
                "module": model.__class__.__module__,
                "params": getattr(model, "get_params", lambda: {})(),
            }
            _atomic_write_json(os.path.join(run_dir, "model_info.json"), model_info)
            paths["model_info"] = os.path.join(run_dir, "model_info.json")
        except Exception:
            pass

    # ---- Save arrays ----
    for key in ("labeled_idx", "labels", "train_idx"):
        arr = result.get(key, None)
        if arr is not None:
            p = os.path.join(run_dir, f"{key}.npy")
            bio = io.BytesIO()
            np.save(bio, np.asarray(arr))
            _atomic_write_bytes(p, bio.getvalue())
            paths[key] = p

    # ---- Save trackers (json + optional parquet) ----
    trackers = result.get("trackers", None)
    if trackers is not None:
        p_json = os.path.join(run_dir, "trackers.json")
        _atomic_write_json(p_json, trackers)
        paths["trackers_json"] = p_json

        if pd is not None:
            try:
                # flatten dict-of-lists into a DataFrame when possible
                df = pd.DataFrame(trackers)
                p_parq = os.path.join(run_dir, "trackers.parquet")
                bio = io.BytesIO()
                df.to_parquet(bio, index=False)
                _atomic_write_bytes(p_parq, bio.getvalue())
                paths["trackers_parquet"] = p_parq
            except Exception:
                pass

    # ---- Save meta ----
    meta_out = {
        "algo_version": result.get("algo_version"),
        "timestamp": ts,
        **meta_in,
    }
    meta_out.setdefault("run_dir", run_dir)
    meta_out.setdefault("model_id", rid)
    meta_out.setdefault("model_run_id", rid)
    if model_family:
        meta_out.setdefault("model_family", model_family)
    if paths.get("model"):
        meta_out.setdefault("model_path", paths["model"])
    trace_in = meta_out.get("trace")
    trace_out = dict(trace_in) if isinstance(trace_in, dict) else {}
    trace_out.setdefault("model_id", meta_out["model_id"])
    trace_out.setdefault("model_run_id", meta_out["model_run_id"])
    if meta_out.get("model_family"):
        trace_out.setdefault("model_family", meta_out["model_family"])
    if meta_out.get("model_path"):
        trace_out.setdefault("model_path", meta_out["model_path"])
    meta_out["trace"] = trace_out
    meta_file = meta_filename or (f"{model_family}_active_meta.json" if model_family else "meta.json")
    p_meta = os.path.join(run_dir, meta_file)
    _atomic_write_json(p_meta, meta_out)
    paths["meta"] = p_meta

    # ---- Human-friendly summary ----
    summary_lines = []
    summary_lines.append(f"Run ID: {rid}")
    summary_lines.append(f"Timestamp: {ts}")
    if "algo_version" in result:
        summary_lines.append(f"Algo version: {result['algo_version']}")
    if meta:
        for k, v in meta.items():
            summary_lines.append(f"{k}: {v}")
    # quick sizes
    for key in ("labeled_idx", "labels", "train_idx"):
        if key in result and result[key] is not None:
            summary_lines.append(f"{key}: {np.asarray(result[key]).shape}")
    _atomic_write_text(os.path.join(run_dir, "SUMMARY.txt"), "\n".join(summary_lines))
    paths["summary"] = os.path.join(run_dir, "SUMMARY.txt")

    return paths

