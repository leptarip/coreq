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

import time
from pathlib import Path
from typing import Optional


def repo_root_from(path: Path) -> Path:
    return next((cand for cand in [path] + list(path.parents) if (cand / "source").is_dir()), path.parent)


def default_surrogate_root(path: Path) -> Path:
    return repo_root_from(path) / "output_surrogate_models"


def prepare_run_dir(root: str | Path, name: str) -> str:
    root_path = Path(root).expanduser()
    parent = root_path / name
    parent.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    base = parent / f"{name}_{ts}"
    candidate = base
    counter = 1
    while candidate.exists():
        candidate = parent / f"{name}_{ts}_{counter:02d}"
        counter += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return str(candidate)


def latest_run_dir(root: str | Path, name: str) -> Optional[Path]:
    parent = Path(root).expanduser() / name
    if not parent.is_dir():
        return None
    runs = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(f"{name}_")]
    if not runs:
        return None
    runs.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return runs[0]
