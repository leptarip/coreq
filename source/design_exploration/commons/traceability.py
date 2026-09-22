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

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_trace(split_path: str | Path, split_manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = Path(split_path).resolve()
    manifest = split_manifest or load_json(p)
    return {
        "split_id": str(manifest.get("split_id") or p.parent.name),
        "dataset_id": str(manifest.get("dataset_id") or p.parent.name),
        "dataset_run_id": str(manifest.get("dataset_run_id") or p.parent.name),
        "split_path": str(p),
        "labels_path": str(manifest.get("labels_path", "")) or None,
        "scenario_name": manifest.get("scenario_name"),
        "space_spec_name": manifest.get("space_spec_name"),
    }

