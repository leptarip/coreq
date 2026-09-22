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

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import joblib
import numpy as np
from scipy.stats import norm as stats_norm
from scipy.stats import t as stats_t

from source.design_exploration.commons.python_paths import (
    get_python311_path,
    require_python311_path,
)
from source.design_exploration.commons.traceability import load_json
from source.design_exploration.surrogates.hetGP import heteroscedastic_gp
from source.design_exploration.surrogates.hetGPt import heteroscedastic_student_t_gp
from source.design_exploration.surrogates.hetGPt.heteroscedastic_student_t_gp import (
    _t_scale_from_variance,
)

logger = logging.getLogger(__name__)

_MODEL_FAMILY_KEYS = ("rf", "nn_ensemble", "hetgp", "hetgpt", "qrf")


class SurrogatePredictor:
    def __init__(
        self,
        name: str,
        per_scenario: Dict[str, Callable[[np.ndarray], np.ndarray]],
    ):
        self.name = name
        self.per_scenario = per_scenario

    def predict_pviol(self, X: np.ndarray) -> np.ndarray:
        if not self.per_scenario:
            raise ValueError(f"Surrogate {self.name} has no scenario predictors.")
        per = [fn(X) for fn in self.per_scenario.values()]
        if not per:
            raise ValueError(f"Surrogate {self.name} produced empty predictions.")
        stacked = np.vstack(per)
        return np.clip(np.sum(stacked, axis=0), 0.0, 1.0)


def _qrf_norm_params(norm_dict: Optional[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not norm_dict:
        return None, None
    g_min = norm_dict.get("g_min")
    g_range = norm_dict.get("g_range")
    g_max = norm_dict.get("g_max")
    if g_min is None:
        return None, None
    g_min_arr = np.asarray(g_min, dtype=float)
    if g_range is None and g_max is not None:
        g_range = np.asarray(g_max, dtype=float) - g_min_arr
    if g_range is None:
        return g_min_arr, None
    g_range_arr = np.asarray(g_range, dtype=float)
    g_range_arr = np.where(g_range_arr > 0, g_range_arr, 1.0)
    return g_min_arr, g_range_arr


def _apply_qrf_norm(X: np.ndarray, g_min: Optional[np.ndarray], g_range: Optional[np.ndarray]) -> np.ndarray:
    if g_min is None or g_range is None:
        return X
    return (X - g_min) / g_range


def _get_python311() -> str:
    return get_python311_path()


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in [here] + list(here.parents):
        if (cand / "source").is_dir():
            return cand
    return here.parent


def _resolved_path(path: str | Path | None) -> Optional[Path]:
    if not path:
        return None
    try:
        return Path(path).resolve()
    except Exception:
        return None


def _same_path(a: str | Path | None, b: str | Path | None) -> bool:
    pa = _resolved_path(a)
    pb = _resolved_path(b)
    return pa is not None and pb is not None and pa == pb


def _infer_model_path_from_meta(meta_path: Path, meta: Dict) -> Optional[str]:
    model_path = meta.get("model_path")
    if model_path:
        return str(Path(model_path).resolve())
    stem = meta_path.stem
    prefix = stem[:-5] if stem.endswith("_meta") else stem
    candidates = sorted(
        [
            p
            for p in meta_path.parent.iterdir()
            if p.is_file() and p.name.startswith(prefix) and p.suffix in {".joblib", ".pkl", ".pickle"}
        ]
    )
    if not candidates:
        return None
    return str(candidates[0].resolve())


def _meta_matches_dataset(
    meta: Dict,
    dataset_manifest_path: Path,
    dataset_manifest: Dict,
    scenario_name: str,
) -> bool:
    if str(meta.get("scenario") or "") != str(scenario_name):
        return False
    manifest_split_path = dataset_manifest.get("split_path") or str(dataset_manifest_path.parent / "offline_splits.json")
    manifest_dataset_id = str(dataset_manifest.get("dataset_id") or dataset_manifest_path.parent.name)
    if _same_path(meta.get("split_path"), manifest_split_path):
        return True
    trace_split = (meta.get("trace") or {}).get("split") or {}
    if _same_path(trace_split.get("split_path"), manifest_split_path):
        return True
    if str(trace_split.get("split_id") or "") == manifest_dataset_id:
        return True
    if str(trace_split.get("dataset_id") or "") == manifest_dataset_id:
        return True
    return False


def _entry_from_model_meta(family: str, meta_path: Path, meta: Dict) -> Optional[Dict]:
    model_path = _infer_model_path_from_meta(meta_path, meta)
    if not model_path:
        return None
    entry: Dict[str, object] = {"model": model_path, "meta": str(meta_path.resolve())}
    if family in {"hetgp", "hetgpt"}:
        if meta.get("bias") is not None:
            entry["bias"] = float(meta["bias"])
        if meta.get("var_scale") is not None:
            entry["var_scale"] = float(meta["var_scale"])
    if family == "hetgpt" and meta.get("df") is not None:
        entry["df"] = float(meta["df"])
    return entry


def discover_surrogate_entries_from_dataset_manifest(
    dataset_manifest_path: str | Path,
    scenario_name: str,
) -> Dict[str, Dict]:
    manifest_path = Path(dataset_manifest_path).resolve()
    dataset_manifest = load_json(manifest_path)
    surrogate_root = manifest_path.parent.parent.parent
    out: Dict[str, Dict] = {}
    for family in _MODEL_FAMILY_KEYS:
        family_dir = surrogate_root / family
        if not family_dir.is_dir():
            continue
        best: Optional[Tuple[float, Dict]] = None
        for meta_path in family_dir.glob("*/*_meta.json"):
            try:
                meta = load_json(meta_path)
            except Exception:
                continue
            if not _meta_matches_dataset(meta, manifest_path, dataset_manifest, scenario_name):
                continue
            entry = _entry_from_model_meta(family, meta_path, meta)
            if not entry:
                continue
            score = meta_path.stat().st_mtime
            if best is None or score > best[0]:
                best = (score, entry)
        if best is not None:
            out[family] = best[1]
    return out


def _run_qrf_subprocess_predict(
    model_path: str,
    meta_path: Optional[str],
    X: np.ndarray,
    taus: np.ndarray,
) -> np.ndarray:
    py311 = require_python311_path(
        _get_python311(),
        cli_flag="--python311",
        purpose="QRF subprocess prediction",
    )
    script_path = _repo_root() / "source" / "design_exploration" / "surrogates" / "qrf" / "predict_qrf_offline.py"
    if not script_path.exists():
        raise FileNotFoundError(f"QRF predict script not found: {script_path}")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        x_path = tmp / "X.npy"
        taus_path = tmp / "taus.npy"
        out_path = tmp / "p_unsafe.npy"
        np.save(x_path, np.asarray(X, dtype=float))
        np.save(taus_path, np.asarray(taus, dtype=float))
        cmd = [
            py311,
            str(script_path),
            "--model",
            model_path,
            "--x-path",
            str(x_path),
            "--taus-path",
            str(taus_path),
            "--out-path",
            str(out_path),
        ]
        if meta_path:
            cmd += ["--meta", meta_path]
        logger.info("[qrf] spawning Python 3.11 for prediction (%s rows)", X.shape[0])
        subprocess.check_call(cmd)
        return np.load(out_path)


def _qrf_predict_p_unsafe(model, X: np.ndarray, taus: np.ndarray) -> np.ndarray:
    qs = np.asarray(taus, dtype=float).reshape(-1)
    qs = np.sort(np.unique(qs))
    preds = np.asarray(model.predict(X, quantiles=qs.tolist()), dtype=float)
    n_samples = X.shape[0]
    n_q = qs.size
    if preds.ndim == 1:
        if n_samples == 1 and preds.shape[0] == n_q:
            preds = preds.reshape(1, n_q)
        elif preds.shape[0] == n_samples:
            preds = preds.reshape(n_samples, 1)
        else:
            raise ValueError(f"Unexpected QRF prediction shape {preds.shape} for {n_samples} samples")
    if preds.ndim == 2:
        if preds.shape == (n_q, n_samples):
            preds = preds.T
        elif preds.shape != (n_samples, n_q) and not (n_samples == 1 and preds.shape[1] == n_q):
            raise ValueError(f"Unexpected QRF prediction shape {preds.shape} for {n_samples} samples and {n_q} qs")
    preds = np.maximum.accumulate(preds, axis=1)
    mask_ge = preds >= 0.0
    any_ge = mask_ge.any(axis=1)
    p = np.full(n_samples, 1.0, dtype=float)
    if np.any(any_ge):
        idx_hi = np.argmax(mask_ge, axis=1)
        rows_any = np.where(any_ge)[0]
        hi = idx_hi[rows_any]
        hi0 = hi == 0
        if np.any(hi0):
            p[rows_any[hi0]] = qs[0]
        sel = hi > 0
        if np.any(sel):
            rows = rows_any[sel]
            hi_sel = hi[sel]
            lo_sel = hi_sel - 1
            q_lo = preds[rows, lo_sel]
            q_hi = preds[rows, hi_sel]
            t_lo = qs[lo_sel]
            t_hi = qs[hi_sel]
            denom = q_hi - q_lo
            frac = np.where(denom > 1e-12, (0.0 - q_lo) / denom, 1.0)
            frac = np.clip(frac, 0.0, 1.0)
            p_val = t_lo + frac * (t_hi - t_lo)
            p[rows] = np.clip(p_val, t_lo, t_hi)
    return np.clip(p, 0.0, 1.0)


def _apply_p_calibration(p: np.ndarray, calib: Optional[Dict]) -> np.ndarray:
    if not calib:
        return p
    xs = np.asarray(calib.get("x", []), dtype=float)
    ys = np.asarray(calib.get("y", []), dtype=float)
    if xs.size == 0 or ys.size == 0:
        return p
    p_cal = np.interp(p, xs, ys, left=ys[0], right=ys[-1])
    tail_alpha = float(calib.get("tail_alpha", 1.0))
    tail_power = float(calib.get("tail_power", 1.0))
    if tail_alpha > 1.0 or tail_power > 1.0:
        p_cal = np.where(p_cal < 0.1, np.minimum(1.0, tail_alpha * np.power(p_cal, 1.0 / tail_power)), p_cal)
    if bool(calib.get("upper_envelope", True)):
        p_cal = np.maximum(p_cal, p)
    return np.clip(p_cal, 0.0, 1.0)


def _predict_classifier(
    model,
    X: np.ndarray,
    unsafe_index: int = 1,
    unsafe_label: int | str = 1,
    invert: bool = False,
) -> np.ndarray:
    def _get_classes(est):
        if hasattr(est, "classes_"):
            return getattr(est, "classes_", None)
        if hasattr(est, "steps") and est.steps:
            last = est.steps[-1][1]
            return getattr(last, "classes_", None)
        return None

    classes = _get_classes(model)
    if classes is not None:
        try:
            col = list(classes).index(unsafe_label)
        except ValueError:
            raise ValueError(
                f"Unsafe label {unsafe_label!r} not found in model classes {list(classes)}; "
                "cannot map probability column safely."
            )
    else:
        col = unsafe_index
    proba = model.predict_proba(X)
    unsafe = np.asarray(proba)[:, col]
    p_viol = unsafe if not invert else (1.0 - unsafe)
    return np.clip(p_viol, 0.0, 1.0)


def _family_predict(family: str, entry: Dict, df_override: Optional[float] = None) -> Callable[[np.ndarray], np.ndarray]:
    """Return X -> P(unsafe) for one trained model of the given family."""
    if family in {"rf", "nn_ensemble"}:
        model = joblib.load(entry["model"])
        return lambda X, m=model: _predict_classifier(
            m, np.asarray(X, dtype=float), unsafe_index=0, unsafe_label=0
        )

    if family == "hetgp":
        art = joblib.load(entry["model"])
        mean_model, var_model = art["mean_model"], art["var_model"]
        norm = (art["norm_lo"], art["norm_span"])
        bias = float(art.get("bias", 0.0))
        var_scale = float(art.get("var_scale", 1.0))

        def _predict_gp(X):
            mu, std_mean, var_ep, _ = heteroscedastic_gp.predict_violation_probability(
                mean_model, var_model, np.asarray(X, dtype=float), norm_params=norm,
            )
            var_total = (np.square(std_mean) + var_ep) * var_scale
            p = stats_norm.cdf((0.0 - (mu + bias)) / np.sqrt(var_total + 1e-12))
            return np.clip(p, 0.0, 1.0)

        return _predict_gp

    if family == "hetgpt":
        art = joblib.load(entry["model"])
        mean_model, var_model = art["mean_model"], art["var_model"]
        norm = (art["norm_lo"], art["norm_span"])
        bias = float(art.get("bias", 0.0))
        var_scale = float(art.get("var_scale", 1.0))
        df = float(df_override if df_override is not None else art.get("df", 5.0))

        def _predict_st(X):
            mu, std_mean, var_ep, _ = heteroscedastic_student_t_gp.predict_violation_probability(
                mean_model, var_model, np.asarray(X, dtype=float), norm_params=norm, df=df,
            )
            var_total = (np.square(std_mean) + var_ep) * var_scale
            scale = _t_scale_from_variance(var_total, df)
            p = stats_t.cdf(0.0, df=df, loc=mu + bias, scale=scale)
            return np.clip(p, 0.0, 1.0)

        return _predict_st

    if family == "qrf":
        model_path = str(entry["model"])
        meta_path = entry.get("meta")
        meta = load_json(Path(meta_path)) if meta_path and Path(meta_path).is_file() else {}
        taus = np.asarray(
            meta.get("cdf_taus") or np.linspace(0.001, 0.999, int(meta.get("cdf_n", 999))),
            dtype=float,
        )
        taus = np.unique(np.clip(taus[np.isfinite(taus)], 1e-6, 1 - 1e-6))
        return lambda X: _run_qrf_subprocess_predict(
            model_path, meta_path, np.asarray(X, dtype=float), taus
        )

    raise ValueError(f"Unsupported surrogate family: {family!r}")


def load_surrogates_from_offline_artifacts(
    artifacts_path: str | Path,
    scenario_name: str,
) -> Dict[str, SurrogatePredictor]:
    """Load every surrogate trained on one offline_builder dataset, via its per-model meta files."""
    entries = discover_surrogate_entries_from_dataset_manifest(artifacts_path, scenario_name)
    out = {
        family: SurrogatePredictor(family, {scenario_name: _family_predict(family, entries[family])})
        for family in _MODEL_FAMILY_KEYS
        if entries.get(family, {}).get("model")
    }
    if not out:
        raise ValueError(f"No surrogates found for scenario '{scenario_name}' from dataset manifest: {artifacts_path}")
    return out


def load_surrogate_family_from_model_metas(
    model_meta_by_scenario: Dict[str, str],
    model_family: Optional[str] = None,
    df_override: Optional[float] = None,
) -> SurrogatePredictor:
    """Load one surrogate family with one model per scenario, from explicit model-meta paths."""
    if not model_meta_by_scenario:
        raise ValueError("model_meta_by_scenario is empty")
    family = str(model_family).strip().lower() if model_family is not None else ""
    per_s: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    for scenario_name, meta_path_str in model_meta_by_scenario.items():
        meta_path = Path(meta_path_str).resolve()
        meta = load_json(meta_path)
        meta_family = str(meta.get("model_family") or family).strip().lower()
        family = family or meta_family
        if family not in _MODEL_FAMILY_KEYS:
            raise ValueError(f"Unsupported surrogate family for explicit model-meta loading: {meta.get('model_family')}")
        if meta_family != family:
            raise ValueError(
                f"Model meta family mismatch for {scenario_name}: expected {family}, got {meta.get('model_family')}"
            )
        entry = _entry_from_model_meta(family, meta_path, meta)
        if not entry:
            raise ValueError(f"Could not infer model artifact from meta: {meta_path}")
        per_s[scenario_name] = _family_predict(family, entry, df_override=df_override)
    return SurrogatePredictor(family, per_s)
