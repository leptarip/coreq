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
# design_space.py
# ===========================
# Design space definitions and utilities
# ===========================
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Union
import itertools
import numpy as np

# ---------------- Spaces ----------------

@dataclass(frozen=True)
class DiscreteSpace:
    """A finite, unordered set of admissible scalar values."""
    values: Tuple[float, ...]


@dataclass(frozen=True)
class LockedTupleSpace:
    """
    A single logical parameter that selects one tuple of length d at a time.
    Example:
        choices=[(A1,A2,A3), (B1,B2,B3), (C1,C2,C3)]
        fields = ("a", "b", "c")
    Will produce flattened keys: "<name>.a", "<name>.b", "<name>.c".
    """
    choices: Tuple[Tuple[float, ...], ...]
    fields: Tuple[str, ...]  # len(fields) == len(choices[0])

# ---------------- Spec & Grid ----------------

@dataclass
class SpaceSpec:
    """Container for a design space specification with automatic key flattening."""
    name: str
    spaces: Dict[str, Union[DiscreteSpace, LockedTupleSpace]]
    _keys: Tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self):
        # Build flattened keys in insertion order of `spaces`
        flat: List[str] = []
        for base, sp in self.spaces.items():
            if isinstance(sp, DiscreteSpace):
                flat.append(base)
            elif isinstance(sp, LockedTupleSpace):
                flat.extend([f"{base}.{f}" for f in sp.fields])
            else:
                raise TypeError(f"Unsupported space type for '{base}': {type(sp)}")
        self._keys = tuple(flat)

    @property
    def keys(self) -> Tuple[str, ...]:
        """Flattened column names (order matches build_grid columns)."""
        return self._keys

    def build_grid(self) -> np.ndarray:
        """
        Cartesian product across spaces.
        - DiscreteSpace contributes a (n,1) block
        - LockedTupleSpace contributes a (n,d) block (locked columns)
        Returns: GRID with shape (∏ sizes, len(self.keys)), dtype=float
        """
        blocks: List[np.ndarray] = []
        # Prepare per-space value blocks
        for base, sp in self.spaces.items():
            if isinstance(sp, DiscreteSpace):
                arr = np.asarray(sp.values, dtype=float)[:, None]     # (n,1)
            elif isinstance(sp, LockedTupleSpace):
                arr = np.asarray(sp.choices, dtype=float)             # (n,d)
            else:
                raise TypeError(f"Unsupported space type for '{base}': {type(sp)}")
            blocks.append(arr)

        # Cartesian product over row indices for each block
        idx_lists = [range(b.shape[0]) for b in blocks]
        rows: List[np.ndarray] = []
        for idxs in itertools.product(*idx_lists):
            pieces = [blocks[i][j] for i, j in enumerate(idxs)]  # each piece: (w_i,)
            rows.append(np.concatenate(pieces, axis=0))
        if rows:
            return np.vstack(rows).astype(float, copy=False)
        return np.zeros((0, len(self.keys)), dtype=float)

# ---------------- Design wrapper ----------------

class Design(dict):
    """A single design point: design-parameter name -> value."""

# ---------------- Design space ----------------
#
# The grid has 27,720 designs. Every artifact the pipeline reads or writes
# (episodes, labels, surrogate scores, audited sets) is indexed by the
# row position in ``SPACE_SPEC.build_grid()`` and records the name "v3".
SPACE: Dict[str, Union[DiscreteSpace, LockedTupleSpace]] = {
    "prob": DiscreteSpace((0.75, 0.975)),
    "gen_t": DiscreteSpace((20, 50, 100)),
    "fault": DiscreteSpace((0, 200, 500, 1000, 2000)),
    "unc_p": DiscreteSpace((-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0)),
    "unc_v": DiscreteSpace((-2.0, -1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 1.5, 2.0)),
    "ego_ad_period": DiscreteSpace((10, 50)),
    "packet_drop_rate": DiscreteSpace((0.01, 0.001)),
    "network": LockedTupleSpace(
        choices=((20.0, 25.0, 10.0), (40.0, 55.0, 50.0), (50.0, 100.0, 100.0)),
        fields=("delay_min", "delay_avg", "jit"),
    ),
}

SPACE_SPEC = SpaceSpec(name="v3", spaces=SPACE)


def get_space_spec(name: str) -> SpaceSpec:
    """Return the design space an artifact declares; only "v3" is supported."""
    if str(name) != SPACE_SPEC.name:
        raise KeyError(f"Unsupported design space {name!r}; expected {SPACE_SPEC.name!r}.")
    return SPACE_SPEC

# ---------------- Utilities ----------------

def normalize_grid(grid: np.ndarray):
    """Return (grid01, lo, span) with feature-wise min-max normalization."""
    lo = grid.min(axis=0)
    hi = grid.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    grid01 = (grid - lo) / span
    return grid01, lo, span
