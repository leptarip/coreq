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
"""Interpreter paths for the Python 3.8 (CARLA) / Python 3.11 (QRF) split.

A path is taken from an explicit argument (usually a CLI flag) or else from the
PYTHON311 / PYTHON38 environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_VARS = {"python311": "PYTHON311", "python38": "PYTHON38"}


def _normalize_path(path: str | None) -> str:
    if path is None:
        return ""
    raw = str(path).strip()
    if not raw or raw.lower() == "none":
        return ""
    return str(Path(raw).expanduser())


def get_python_path(key: str, explicit: str | None = None) -> str:
    norm = str(key).strip().lower()
    path = _normalize_path(explicit)
    if path:
        return path

    if norm not in _ENV_VARS:
        raise ValueError(f"Unsupported Python path key: {key!r}")
    return _normalize_path(os.environ.get(_ENV_VARS[norm]))


def require_python_path(
    key: str,
    explicit: str | None = None,
    *,
    cli_flag: str | None = None,
    purpose: str | None = None,
) -> str:
    norm = str(key).strip().lower()
    path = get_python_path(norm, explicit)
    label = "Python 3.11" if norm == "python311" else "Python 3.8"
    setup_parts = [f"set the {_ENV_VARS[norm]} environment variable"]
    if cli_flag:
        setup_parts.insert(0, f"pass {cli_flag}")
    if len(setup_parts) == 1:
        setup_msg = setup_parts[0]
    else:
        setup_msg = f"{', '.join(setup_parts[:-1])}, or {setup_parts[-1]}"

    if not path:
        detail = f" It is required for {purpose}." if purpose else ""
        raise RuntimeError(
            f"{label} interpreter path is not configured.{detail} "
            f"To fix this, {setup_msg}."
        )

    if not Path(path).exists():
        detail = f" It is required for {purpose}." if purpose else ""
        raise FileNotFoundError(
            f"{label} interpreter path does not exist: {path}.{detail} "
            f"Update it by {setup_msg}."
        )

    return path


def get_python311_path(explicit: str | None = None) -> str:
    return get_python_path("python311", explicit)


def get_python38_path(explicit: str | None = None) -> str:
    return get_python_path("python38", explicit)


def require_python311_path(
    explicit: str | None = None,
    *,
    cli_flag: str | None = None,
    purpose: str | None = None,
) -> str:
    return require_python_path("python311", explicit, cli_flag=cli_flag, purpose=purpose)


def require_python38_path(
    explicit: str | None = None,
    *,
    cli_flag: str | None = None,
    purpose: str | None = None,
) -> str:
    return require_python_path("python38", explicit, cli_flag=cli_flag, purpose=purpose)
