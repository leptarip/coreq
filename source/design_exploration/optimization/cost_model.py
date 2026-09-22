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
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from source.design_exploration.commons.design_space import Design, SpaceSpec


ENGINEERING_COST_MODEL_ID = "illustrative_engineering_cost_v1"


def _number_key(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    return format(value, ".15g")


def validate_engineering_cost_config(
    config: Mapping[str, Any], space_spec: SpaceSpec
) -> None:
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("cost profile schema_version must be 1")
    if config.get("model_id") != ENGINEERING_COST_MODEL_ID:
        raise ValueError(
            "cost profile model_id must be %r" % ENGINEERING_COST_MODEL_ID
        )
    base_cost = float(config.get("base_cost", 0.0))
    if not math.isfinite(base_cost) or base_cost <= 0.0:
        raise ValueError("base_cost must be finite and positive")

    components = list(config.get("components", []))
    if not components:
        raise ValueError("at least one cost component is required")
    component_ids = [str(component.get("id", "")) for component in components]
    if any(not item for item in component_ids) or len(set(component_ids)) != len(
        component_ids
    ):
        raise ValueError("component ids must be present and unique")

    known_keys = set(space_spec.keys)
    known_subsystems = set()
    covered_keys = []
    for component in components:
        keys = tuple(component.get("keys", []))
        if not keys or not set(keys).issubset(known_keys):
            raise ValueError(
                "component %r has unknown design keys" % component.get("id")
            )
        covered_keys.extend(keys)
        transform = component.get("transform")
        if transform not in ("identity", "absolute", "tuple"):
            raise ValueError(
                "component %r has invalid transform" % component.get("id")
            )
        premiums = component.get("premiums")
        if transform == "tuple":
            if not isinstance(premiums, list) or not premiums:
                raise ValueError(
                    "tuple component %r needs a non-empty premium list"
                    % component.get("id")
                )
            if any(len(row.get("values", [])) != len(keys) for row in premiums):
                raise ValueError(
                    "tuple component %r has a tuple-width mismatch"
                    % component.get("id")
                )
            premium_values = [float(row["premium"]) for row in premiums]
        else:
            if not isinstance(premiums, dict) or not premiums:
                raise ValueError(
                    "scalar component %r needs a non-empty premium map"
                    % component.get("id")
                )
            if len(keys) != 1:
                raise ValueError("scalar transform requires exactly one key")
            premium_values = [float(value) for value in premiums.values()]
        if any(
            not math.isfinite(value) or value < 0.0 for value in premium_values
        ):
            raise ValueError(
                "component %r premiums must be finite and non-negative"
                % component.get("id")
            )
        subsystem = str(component.get("subsystem", ""))
        if not subsystem:
            raise ValueError("every cost component needs a subsystem")
        known_subsystems.add(subsystem)
    if set(covered_keys) != known_keys or len(covered_keys) != len(known_keys):
        raise ValueError(
            "cost components must cover every design key exactly once"
        )

    profiles = config.get("profiles", {})
    if "central" not in profiles:
        raise ValueError("a central profile is required")
    for profile_id, profile in profiles.items():
        subsystem_multipliers = profile.get("subsystem_multipliers", {})
        if set(subsystem_multipliers) != known_subsystems:
            raise ValueError(
                "profile %r must cover every subsystem exactly" % profile_id
            )
        values = [float(value) for value in subsystem_multipliers.values()]
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(
                "profile subsystem multipliers must be finite and positive"
            )
        component_multipliers = profile.get("component_multipliers", {})
        unknown_components = set(component_multipliers).difference(component_ids)
        if unknown_components:
            raise ValueError(
                "profile %r has unknown component multipliers: %s"
                % (profile_id, sorted(unknown_components))
            )
        component_values = [
            float(value) for value in component_multipliers.values()
        ]
        if any(
            not math.isfinite(value) or value < 0.0
            for value in component_values
        ):
            raise ValueError(
                "profile component multipliers must be finite and non-negative"
            )

    sensor_bias_semantics = config.get("sensor_bias_semantics", {})
    required_sensor_bias_semantics = {
        "meaning": "fixed_signed_residual_systematic_measurement_bias",
        "residual_boundary": (
            "after_all_calibration_and_compensation_represented_by_the_design"
        ),
        "design_identity": "signed_unc_p_unc_v_tuple",
        "cost_basis": "absolute_residual_bias_grade",
        "sign_cost_policy": "equal_magnitude_equal_premium",
        "performance_separation": "cost_does_not_encode_behavioral_harm",
    }
    for key, expected in required_sensor_bias_semantics.items():
        if sensor_bias_semantics.get(key) != expected:
            raise ValueError(
                "sensor_bias_semantics.%s must be %r" % (key, expected)
            )

    nomenclature = config.get("nomenclature", {})
    if nomenclature.get("stored_design_key") != "fault":
        raise ValueError("the historical stored design key must remain 'fault'")
    if (
        nomenclature.get("scientific_name")
        != "guaranteed_service_interruption_ms"
    ):
        raise ValueError("the recovery coordinate needs the agreed scientific name")

    hv_policy = config.get("hypervolume_policy") or {}
    cost_reference = hv_policy.get("cost_reference") or {}
    if cost_reference.get("method") != "ceil_candidate_max_multiplier":
        raise ValueError(
            "hypervolume cost reference must use ceil_candidate_max_multiplier"
        )
    multiplier = float(cost_reference.get("multiplier", float("nan")))
    if not math.isfinite(multiplier) or multiplier <= 1.0:
        raise ValueError("hypervolume cost-reference multiplier must exceed one")
    if cost_reference.get("population") != "candidate_set":
        raise ValueError(
            "hypervolume cost-reference population must be candidate_set"
        )
    performance_reference = float(
        hv_policy.get("performance_reference", float("nan"))
    )
    if not math.isfinite(performance_reference):
        raise ValueError("hypervolume performance_reference must be finite")
    if hv_policy.get("comparability") != "within_active_cost_profile_only":
        raise ValueError(
            "hypervolume comparability must be within_active_cost_profile_only"
        )


def engineering_component_costs(
    design: Mapping[str, float],
    config: Mapping[str, Any],
    profile_id: str,
) -> Dict[str, float]:
    if profile_id not in config["profiles"]:
        raise ValueError("unknown cost profile %r" % profile_id)
    profile = config["profiles"][profile_id]
    subsystem_multipliers = profile["subsystem_multipliers"]
    component_multipliers = profile.get("component_multipliers", {})
    result: Dict[str, float] = {}
    for component in config["components"]:
        keys = tuple(component["keys"])
        transform = component["transform"]
        if transform == "tuple":
            observed = tuple(float(design[key]) for key in keys)
            matches = [
                row
                for row in component["premiums"]
                if tuple(float(value) for value in row["values"]) == observed
            ]
            if len(matches) != 1:
                raise ValueError(
                    "no unique premium for %s=%r" % (component["id"], observed)
                )
            base = float(matches[0]["premium"])
        else:
            value = float(design[keys[0]])
            if transform == "absolute":
                value = abs(value)
            premium_key = _number_key(value)
            if premium_key not in component["premiums"]:
                raise ValueError(
                    "no premium for %s=%s" % (component["id"], premium_key)
                )
            base = float(component["premiums"][premium_key])
        component_id = str(component["id"])
        result[component_id] = (
            base
            * float(subsystem_multipliers[component["subsystem"]])
            * float(component_multipliers.get(component_id, 1.0))
        )
    return result


def evaluate_engineering_cost(
    design: Mapping[str, float],
    config: Mapping[str, Any],
    profile_id: str = "central",
) -> float:
    return float(config["base_cost"]) + sum(
        engineering_component_costs(design, config, profile_id).values()
    )


class EngineeringCostModel:
    """Frozen table-driven engineering-cost model over the V3 design space."""

    model_id = ENGINEERING_COST_MODEL_ID

    def __init__(
        self, space_spec: SpaceSpec, config_path: Path, profile_id: str = "central"
    ):
        self.space_spec = space_spec
        self.keys = space_spec.keys
        self.config_path = Path(config_path).resolve()
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = json.load(handle)
        validate_engineering_cost_config(self.config, space_spec)
        if profile_id not in self.config["profiles"]:
            raise ValueError("unknown cost profile %r" % profile_id)
        self.profile_id = str(profile_id)

        # Every V3 design must receive a valid cost.
        maximum_cost = float(self.config["base_cost"])
        for row in space_spec.build_grid():
            design = Design(zip(space_spec.keys, (float(value) for value in row)))
            value = self.evaluate(design)
            if not math.isfinite(value) or value < float(self.config["base_cost"]):
                raise ValueError("cost profile emitted an invalid full-grid value")
            maximum_cost = max(maximum_cost, value)
        maximum_ratio = maximum_cost / float(self.config["base_cost"])
        configured_bound = float(
            (self.config.get("bounds") or {}).get(
                "maximum_total_to_base_ratio", float("nan")
            )
        )
        if not math.isfinite(configured_bound) or maximum_ratio > configured_bound:
            raise ValueError("cost profile exceeds its declared full-system ratio bound")

    def evaluate(self, design: Design) -> float:
        return evaluate_engineering_cost(design, self.config, self.profile_id)


def build_cost_model(
    space_spec: SpaceSpec,
    *,
    config_path: str,
    profile_id: str = "central",
) -> "EngineeringCostModel":
    """Engineering-cost model for one profile of a cost-profile JSON."""
    if not config_path:
        raise ValueError("the engineering cost model requires a cost profile JSON")
    return EngineeringCostModel(space_spec, Path(config_path), profile_id)


def cost_model_manifest(model, space_spec: SpaceSpec) -> Dict[str, Any]:
    """Describe the cost model and its cost range over the design grid."""
    costs = np.asarray(
        [model.evaluate(Design(zip(space_spec.keys, (float(v) for v in row))))
         for row in space_spec.build_grid()],
        dtype=float,
    )
    return {
        "model_id": str(model.model_id),
        "profile_id": model.profile_id,
        "config_path": str(model.config_path),
        "space_spec_name": str(space_spec.name),
        "cost_min": float(costs.min()),
        "cost_max": float(costs.max()),
    }


def resolve_hypervolume_reference(
    model: EngineeringCostModel,
    space_spec: SpaceSpec,
    candidate_indices: np.ndarray,
) -> Dict[str, Any]:
    """Resolve the profile's hypervolume rule over the candidate set."""
    if not isinstance(model, EngineeringCostModel):
        raise ValueError(
            "profile-derived hypervolume references require EngineeringCostModel"
        )
    indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    grid = space_spec.build_grid()
    if indices.size == 0:
        raise ValueError("cannot derive a hypervolume reference from no candidates")
    if np.any(indices < 0) or np.any(indices >= len(grid)):
        raise ValueError("hypervolume candidate indices are outside the design grid")
    if not np.array_equal(indices, np.unique(indices)):
        raise ValueError("hypervolume candidate indices must be sorted and unique")

    costs = np.asarray(
        [
            model.evaluate(
                Design(
                    zip(
                        space_spec.keys,
                        (float(value) for value in grid[int(index)]),
                    )
                )
            )
            for index in indices
        ],
        dtype=float,
    )
    policy = model.config["hypervolume_policy"]
    cost_policy = policy["cost_reference"]
    multiplier = float(cost_policy["multiplier"])
    candidate_max = float(costs.max())
    unrounded = multiplier * candidate_max
    resolved_cost = float(math.ceil(unrounded))
    if resolved_cost <= candidate_max:
        raise ValueError("resolved hypervolume cost reference is not worse than all candidates")
    resolved_perf = float(policy["performance_reference"])
    return {
        "source": "cost_profile_policy",
        "cost_profile_id": model.profile_id,
        "cost_reference": {
            "method": str(cost_policy["method"]),
            "multiplier": multiplier,
            "population": str(cost_policy["population"]),
            "candidate_count": int(indices.size),
            "candidate_cost_min": float(costs.min()),
            "candidate_cost_max": candidate_max,
            "unrounded_reference": float(unrounded),
            "resolved_reference": resolved_cost,
        },
        "performance_reference": resolved_perf,
        "comparability": str(policy["comparability"]),
        "resolved_point": [resolved_cost, resolved_perf],
    }
