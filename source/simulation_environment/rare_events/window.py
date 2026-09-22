# Copyright (c) 2025 278097159+leptarip@users.noreply.github.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pre-draw conditioning helpers for temporally localized proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class ConditionalBiasWindow:
    start_ms: int
    end_ms: int
    until_successes: Optional[int] = None


BiasWindow = Optional[ConditionalBiasWindow]


def parse_bias_window(bias_cfg: Optional[Mapping]) -> BiasWindow:
    """Return an inclusive ``(start_ms, end_ms)`` proposal window.

    A missing window preserves the historical globally active proposal.  The
    simulator time is known before each sensor/network draw, so selecting the
    conditional proposal from this window does not depend on the outcome being
    sampled and retains exact sequential likelihood accounting.
    """
    if not bias_cfg or "active_window" not in bias_cfg:
        return None
    window = bias_cfg["active_window"]
    if not isinstance(window, Mapping):
        raise ValueError("active_window must be a mapping")
    start = int(window["start_ms"])
    end = int(window["end_ms"])
    if start < 0 or end < start:
        raise ValueError("active_window requires 0 <= start_ms <= end_ms")
    until_successes = window.get("until_successes")
    if until_successes is not None:
        until_successes = int(until_successes)
        if until_successes < 1:
            raise ValueError("active_window.until_successes must be positive")
    return ConditionalBiasWindow(start, end, until_successes)


def bias_window_contains_time(window: BiasWindow, sim_time) -> bool:
    """Whether simulator time is inside the configured inclusive interval."""
    if window is None:
        return True
    if sim_time is None:
        raise ValueError("sim_time is required by an active_window proposal")
    return window.start_ms <= int(sim_time) <= window.end_ms


def bias_window_active(window: BiasWindow, sim_time, successes: int = 0) -> bool:
    """Whether a conditional proposal is active for this pre-draw state."""
    if window is None:
        return True
    if not bias_window_contains_time(window, sim_time):
        return False
    return window.until_successes is None or int(successes) < window.until_successes


def validate_bias_window(bias_cfg: Mapping, path: str):
    """Return the project's conventional ``(valid, reason)`` result."""
    if "active_window" not in bias_cfg:
        return True, ""
    window = bias_cfg["active_window"]
    if not isinstance(window, Mapping):
        return False, f"{path}.active_window must be a mapping"
    required = {"start_ms", "end_ms"}
    allowed = required | {"until_successes"}
    if not required.issubset(window) or not set(window).issubset(allowed):
        return False, (
            f"{path}.active_window requires start_ms/end_ms and optionally "
            "until_successes"
        )
    try:
        start = int(window["start_ms"])
        end = int(window["end_ms"])
    except (TypeError, ValueError):
        return False, f"{path}.active_window bounds must be integers"
    if start < 0 or end < start:
        return False, f"{path}.active_window requires 0 <= start_ms <= end_ms"
    if "until_successes" in window:
        try:
            successes = int(window["until_successes"])
        except (TypeError, ValueError):
            return False, f"{path}.active_window.until_successes must be an integer"
        if successes < 1:
            return False, f"{path}.active_window.until_successes must be positive"
    return True, ""
