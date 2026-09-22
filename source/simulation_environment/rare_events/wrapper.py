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
Lightweight importance-sampling wrapper for rare-event estimation.

Usage (example):
    sampler = ImportanceSampler(seed=0)
    miss = sampler.bernoulli(p_base=0.01, p_bias=0.1)
    delay_ms = sampler.lognormal(mu_base, sigma_base, mu_bias, sigma_bias)
    final_weight = sampler.weight()

The class keeps track of the running log-likelihood ratio between the base
distribution (the one you want to estimate under) and the biased distribution
you actually sample from. You can plug this into the simulator by replacing
calls to random.* with the corresponding sampler methods.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from source.simulation_environment.configuration_error import SimulationConfigurationError

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def require_sampler_for_bias(sampler, bias_cfg, component: str) -> None:
    """Reject an active proposal configuration without weight bookkeeping."""
    if bias_cfg and sampler is None:
        raise SimulationConfigurationError(
            f"{component} bias configuration requires an ImportanceSampler; "
            "sampler-less execution must remain nominal."
        )


def replay_configuration_is_nominal(
    cfg: Mapping[str, Any], re_cfg: Mapping[str, Any],
) -> bool:
    """Verify from simulator parameters that a replay proposal is the nominal law."""
    sensor = re_cfg.get("sensor_bias", {}) or {}
    if any((
        float(sensor.get("miss_scale", 1.0)) != 1.0,
        float(sensor.get("unc_p_add", 0.0)) != 0.0,
        float(sensor.get("unc_v_add", 0.0)) != 0.0,
    )):
        return False

    fault = re_cfg.get("fault_bias", {}) or {}
    base_fault = cfg.get("communication", {}).get("fault", {}) or {}
    if "trigger" in fault:
        return False
    if (
        float(fault.get("trigger_mean_add", 0.0)) != 0.0
        or float(fault.get("trigger_var_scale", 1.0)) != 1.0
    ):
        return False
    for key in ("trigger_mean", "trigger_var"):
        if key in fault:
            base_value = base_fault.get(key)
            if base_value is None or float(fault[key]) != float(base_value):
                return False

    network = re_cfg.get("network_bias", {}) or {}
    if not network:
        return True
    base_network = cfg.get("communication", {}).get("network", {}) or {}
    network_type = base_network.get("type")
    if network_type == "log_norm":
        parameters = ("delay_min", "delay_avg", "jitter", "packet_drop_rate")
    elif network_type == "uniform":
        parameters = ("delay", "jitter", "packet_drop_rate")
    else:
        return False
    return all(
        key not in network or float(network[key]) == float(base_network.get(key))
        for key in parameters
    )


def _validate_prob(p: float, name: str = "probability") -> float:
    value = float(p)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return value


def _normalize_probs(probs: Sequence[float]) -> Tuple[float, ...]:
    vals = [float(p) for p in probs]
    if not vals:
        raise ValueError("Probability list is empty.")
    if any(p < 0.0 for p in vals):
        raise ValueError("Probabilities must be non-negative.")
    total = sum(vals)
    if total <= 0:
        raise ValueError("Probability list must have a positive sum.")
    return tuple(v / total for v in vals)


def _sample_categorical(rng: random.Random, probs: Sequence[float]) -> int:
    r = rng.random()
    acc = 0.0
    last_positive = None
    for idx, p in enumerate(probs):
        if p <= 0.0:
            continue
        last_positive = idx
        acc += p
        if r < acc:
            return idx
    if last_positive is None:
        raise ValueError("Categorical distribution must have positive mass.")
    return last_positive  # numerical guard without selecting a zero-mass tail


def _logpdf_normal(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("Normal sigma must be > 0.")
    z = (x - mu) / sigma
    return -math.log(sigma * _SQRT_2PI) - 0.5 * z * z


def _logpdf_lognormal(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("Log-normal sigma must be > 0.")
    if x <= 0:
        return -math.inf
    logx = math.log(x)
    z = (logx - mu) / sigma
    return -math.log(x * sigma * _SQRT_2PI) - 0.5 * z * z


def defensive_mixture_log_weight(component_log_ratio: float, alpha: float) -> float:
    """Convert ``log(p/q_bias)`` to ``log(p/(alpha*p+(1-alpha)*q_bias))``."""
    mixture_probability = _validate_prob(alpha, "defensive_mixture_probability")
    log_ratio = float(component_log_ratio)
    if mixture_probability == 0.0:
        return log_ratio
    if mixture_probability == 1.0:
        return 0.0
    if log_ratio == -math.inf:
        return -math.inf
    if log_ratio == math.inf:
        return -math.log(mixture_probability)
    log_alpha_ratio = math.log(mixture_probability) + log_ratio
    log_biased = math.log1p(-mixture_probability)
    shift = max(log_alpha_ratio, log_biased)
    log_denominator = shift + math.log(
        math.exp(log_alpha_ratio - shift) + math.exp(log_biased - shift)
    )
    return log_ratio - log_denominator


def finite_mixture_log_weight(
    proposal_log_ratios: Mapping[str, float],
    mixture_probabilities: Mapping[str, float],
) -> float:
    """Return ``log(p / sum_j alpha_j q_j)`` from ``log(p/q_j)`` values."""
    if set(proposal_log_ratios) != set(mixture_probabilities):
        raise ValueError("Proposal ratios and mixture probabilities must have identical keys.")
    probabilities = {name: float(value) for name, value in mixture_probabilities.items()}
    if any(value < 0.0 or not math.isfinite(value) for value in probabilities.values()):
        raise ValueError("Mixture probabilities must be finite and non-negative.")
    total_probability = sum(probabilities.values())
    if not math.isclose(total_probability, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Mixture probabilities must sum to one.")
    terms = []
    for name, alpha in probabilities.items():
        if alpha == 0.0:
            continue
        log_ratio = float(proposal_log_ratios[name])
        if math.isnan(log_ratio):
            raise ValueError("Proposal log ratios may not be NaN.")
        terms.append(math.log(alpha) - log_ratio)
    if not terms:
        raise ValueError("At least one mixture component must have positive probability.")
    shift = max(terms)
    if shift == math.inf:
        return -math.inf
    if shift == -math.inf:
        return math.inf
    return -(shift + math.log(sum(math.exp(value - shift) for value in terms)))


@dataclass
class ImportanceSampler:
    """
    Tracks the log-likelihood ratio between a base distribution and the biased
    distribution actually sampled during simulation.
    """

    seed: int | None = None
    defensive_mixture_probability: float = 0.0
    record_draw_history: bool = False
    audit_proposal_ids: Sequence[str] = field(default_factory=tuple)
    draw_count: int = 0
    rng: random.Random = field(init=False)
    proposal_component: str = field(init=False, default="biased")
    _component_log_ratio: float = field(init=False, default=0.0)
    _component_log_ratios: Dict[str, float] = field(init=False, default_factory=dict)
    _component_draw_counts: Dict[str, int] = field(init=False, default_factory=dict)
    _context_time_ms: Optional[int] = field(init=False, default=None)
    _draw_history: List[Dict[str, Any]] = field(init=False, default_factory=list)
    _audit_proposal_log_ratios: Dict[str, float] = field(init=False, default_factory=dict)
    _audit_proposal_component_log_ratios: Dict[str, Dict[str, float]] = field(
        init=False, default_factory=dict
    )
    _replay_prefix: List[Dict[str, Any]] = field(init=False, default_factory=list)
    _replay_index: int = field(init=False, default=0)

    def __post_init__(self):
        self.defensive_mixture_probability = _validate_prob(
            self.defensive_mixture_probability, "defensive_mixture_probability"
        )
        self.rng = random.Random(self.seed)
        self.audit_proposal_ids = tuple(str(value) for value in self.audit_proposal_ids)
        if any(not value.strip() for value in self.audit_proposal_ids):
            raise ValueError("Audit proposal identifiers must be non-empty.")
        if len(set(self.audit_proposal_ids)) != len(self.audit_proposal_ids):
            raise ValueError("Audit proposal identifiers must be unique.")
        self.reset(self.seed)

    # --------- bookkeeping ---------

    def reset(
        self, seed: int | None = None,
        replay_prefix: Optional[Sequence[Mapping[str, Any]]] = None,
        *, replay_under_nominal_law: bool = False,
    ) -> None:
        """Start an episode, optionally replaying a declared-nominal exact prefix."""
        self._component_log_ratio = 0.0
        self._component_log_ratios = {}
        self._component_draw_counts = {}
        self.draw_count = 0
        self._context_time_ms = None
        self._draw_history = []
        self._audit_proposal_log_ratios = {
            proposal_id: 0.0 for proposal_id in self.audit_proposal_ids
        }
        self._audit_proposal_component_log_ratios = {
            proposal_id: {} for proposal_id in self.audit_proposal_ids
        }
        self._replay_prefix = [dict(value) for value in (replay_prefix or [])]
        self._replay_index = 0
        if self._replay_prefix and not replay_under_nominal_law:
            raise SimulationConfigurationError(
                "Stochastic-prefix replay requires an explicit nominal-law declaration; "
                "replay under an importance-biased proposal is invalid."
            )
        for index, record in enumerate(self._replay_prefix):
            if int(record.get("index", -1)) != index:
                raise SimulationConfigurationError(
                    "Replay-prefix records must have contiguous zero-based indices."
                )
        if seed is not None:
            self.seed = int(seed)
            self.rng.seed(self.seed)
        if self.defensive_mixture_probability == 0.0:
            self.proposal_component = "biased"
        elif self.defensive_mixture_probability == 1.0:
            self.proposal_component = "nominal"
        else:
            self.proposal_component = (
                "nominal"
                if self.rng.random() < self.defensive_mixture_probability
                else "biased"
            )

    @property
    def component_log_ratio(self) -> float:
        """Return ``log(p/q_bias)`` for the realized episode path."""
        return self._component_log_ratio

    @property
    def component_log_ratios(self) -> Dict[str, float]:
        """Return named contributions whose sum is ``component_log_ratio``."""
        return dict(self._component_log_ratios)

    @property
    def component_draw_counts(self) -> Dict[str, int]:
        """Return the number of weighted draws recorded for each mechanism."""
        return dict(self._component_draw_counts)

    @property
    def draw_history(self) -> List[Dict[str, Any]]:
        """Return JSON-safe stochastic draws with their simulator-time context."""
        return [dict(value) for value in self._draw_history]

    @property
    def audit_proposal_log_ratios(self) -> Dict[str, float]:
        """Return ``log(p/q_j)`` for every cross-evaluated proposal ``q_j``."""
        return dict(self._audit_proposal_log_ratios)

    @property
    def audit_proposal_component_log_ratios(self) -> Dict[str, Dict[str, float]]:
        """Return mechanism-level contributions for each cross-evaluated proposal."""
        return {
            proposal_id: dict(values)
            for proposal_id, values in self._audit_proposal_component_log_ratios.items()
        }

    @property
    def replay_prefix_length(self) -> int:
        return len(self._replay_prefix)

    @property
    def replay_prefix_consumed(self) -> bool:
        return self._replay_index == len(self._replay_prefix)

    def set_context_time_ms(self, sim_time: Optional[int]) -> None:
        """Attach simulator time to subsequent draws for replay split points."""
        self._context_time_ms = None if sim_time is None else int(sim_time)

    @property
    def log_weight(self) -> float:
        """Return the exact defensive-mixture ratio ``log(p/q_mix)``.

        The episode proposal is ``q_mix = alpha*p + (1-alpha)*q_bias``.
        Tracking ``p/q_bias`` is sufficient to evaluate the mixture ratio for
        paths drawn from either component.  For ``alpha > 0`` the resulting
        weight is bounded above by ``1/alpha``.
        """
        return defensive_mixture_log_weight(
            self._component_log_ratio, self.defensive_mixture_probability
        )

    def _sample_base_component(self) -> bool:
        return self.proposal_component == "nominal"

    def _sample_or_replay(
        self, kind: str, component: str, fresh: Callable[[], Any],
    ) -> Tuple[Any, bool]:
        if self._replay_index >= len(self._replay_prefix):
            return fresh(), False
        record = self._replay_prefix[self._replay_index]
        expected_time = record.get("context_time_ms")
        if (
            record.get("kind") != kind
            or record.get("component") != component
            or expected_time != self._context_time_ms
        ):
            raise SimulationConfigurationError(
                "Stochastic replay diverged at draw "
                f"{self._replay_index}: expected "
                f"{record.get('kind')}/{record.get('component')}@{expected_time}, got "
                f"{kind}/{component}@{self._context_time_ms}."
            )
        self._replay_index += 1
        return record.get("value"), True

    def _record_log_ratio(
        self, value: float, component: str, *, kind: str,
        sampled_value: Any, replayed: bool,
    ) -> None:
        name = str(component).strip()
        if not name:
            raise ValueError("Likelihood contribution component must be non-empty.")
        contribution = float(value)
        updated_total = self._component_log_ratio + contribution
        updated_component = self._component_log_ratios.get(name, 0.0) + contribution
        if (
            math.isnan(contribution)
            or math.isnan(updated_total)
            or math.isnan(updated_component)
        ):
            raise SimulationConfigurationError(
                "Likelihood-ratio accumulation became NaN; conflicting infinite "
                "contributions indicate incompatible nominal/proposal supports."
            )
        self._component_log_ratio = updated_total
        self._component_log_ratios[name] = updated_component
        self._component_draw_counts[name] = self._component_draw_counts.get(name, 0) + 1
        if self.record_draw_history:
            self._draw_history.append({
                "index": self.draw_count,
                "kind": str(kind),
                "component": name,
                "context_time_ms": self._context_time_ms,
                "value": sampled_value,
                "source": "replayed" if replayed else "fresh",
            })
        self.draw_count += 1

    def _record_audit_log_ratios(
        self, values: Mapping[str, float], component: str,
    ) -> None:
        expected = set(self.audit_proposal_ids)
        if set(values) != expected:
            raise SimulationConfigurationError(
                "Cross-proposal likelihood keys differ from the declared audit proposals."
            )
        for proposal_id, raw_value in values.items():
            contribution = float(raw_value)
            old_total = self._audit_proposal_log_ratios[proposal_id]
            old_component = self._audit_proposal_component_log_ratios[proposal_id].get(
                component, 0.0
            )
            new_total = old_total + contribution
            new_component = old_component + contribution
            if any(math.isnan(value) for value in (contribution, new_total, new_component)):
                raise SimulationConfigurationError(
                    "Cross-proposal likelihood accumulation became NaN."
                )
            self._audit_proposal_log_ratios[proposal_id] = new_total
            self._audit_proposal_component_log_ratios[proposal_id][component] = new_component

    def weight(self) -> float:
        """Return exp(log_weight)."""
        if self.log_weight == -math.inf:
            return 0.0
        if self.log_weight > math.log(sys.float_info.max):
            return math.inf
        return math.exp(self.log_weight)

    # --------- sampling primitives ---------

    def bernoulli(
        self, p_base: float, p_bias: float, *, component: str = "unlabeled",
        audit_probs_bias: Optional[Mapping[str, float]] = None,
    ) -> bool:
        """
        Draw a Bernoulli under the biased probability and update the log-weight.
        """
        p_biased = _validate_prob(p_bias, "p_bias")
        p_ref = _validate_prob(p_base, "p_base")
        if p_ref > 0.0 and p_biased == 0.0:
            raise ValueError("Biased Bernoulli distribution excludes a base-supported true outcome.")
        if p_ref < 1.0 and p_biased == 1.0:
            raise ValueError("Biased Bernoulli distribution excludes a base-supported false outcome.")
        p_sample = p_ref if self._sample_base_component() else p_biased
        outcome, replayed = self._sample_or_replay(
            "bernoulli", component, lambda: self.rng.random() < p_sample
        )
        if replayed and p_ref != p_biased:
            raise SimulationConfigurationError(
                "Stochastic-prefix replay is allowed only under the nominal law."
            )
        if not isinstance(outcome, bool):
            raise SimulationConfigurationError("Replayed Bernoulli value must be boolean.")
        prob_bias = p_biased if outcome else 1.0 - p_biased
        prob_base = p_ref if outcome else 1.0 - p_ref
        log_p_bias = -math.inf if prob_bias == 0.0 else math.log(prob_bias)
        log_p_base = -math.inf if prob_base == 0.0 else math.log(prob_base)

        self._record_log_ratio(
            log_p_base - log_p_bias, component, kind="bernoulli",
            sampled_value=outcome, replayed=replayed,
        )
        if self.audit_proposal_ids:
            alternatives = audit_probs_bias or {}
            audit_values = {}
            for proposal_id in self.audit_proposal_ids:
                p_alt = _validate_prob(
                    alternatives.get(proposal_id, p_ref),
                    f"audit probability for {proposal_id}",
                )
                prob_alt = p_alt if outcome else 1.0 - p_alt
                log_p_alt = -math.inf if prob_alt == 0.0 else math.log(prob_alt)
                audit_values[proposal_id] = log_p_base - log_p_alt
            self._record_audit_log_ratios(audit_values, component)
        return outcome

    def categorical(
        self, probs_base: Sequence[float], probs_bias: Sequence[float],
        *, component: str = "unlabeled",
        audit_probs_bias: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> int:
        """
        Draw an index from a categorical distribution and update the log-weight.
        """
        if len(probs_base) != len(probs_bias):
            raise ValueError("probs_base and probs_bias must have the same length.")

        probs_b = _normalize_probs(probs_bias)
        probs_a = _normalize_probs(probs_base)
        if any(p_base > 0.0 and p_bias == 0.0 for p_base, p_bias in zip(probs_a, probs_b)):
            raise ValueError("Biased categorical distribution excludes a base-supported outcome.")
        probs_sample = probs_a if self._sample_base_component() else probs_b
        idx, replayed = self._sample_or_replay(
            "categorical", component,
            lambda: _sample_categorical(self.rng, probs_sample),
        )
        if replayed and probs_a != probs_b:
            raise SimulationConfigurationError(
                "Stochastic-prefix replay is allowed only under the nominal law."
            )
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(probs_a):
            raise SimulationConfigurationError("Replayed categorical index is invalid.")

        log_p_base = -math.inf if probs_a[idx] == 0.0 else math.log(probs_a[idx])
        log_p_bias = -math.inf if probs_b[idx] == 0.0 else math.log(probs_b[idx])
        self._record_log_ratio(
            log_p_base - log_p_bias, component, kind="categorical",
            sampled_value=idx, replayed=replayed,
        )
        if self.audit_proposal_ids:
            alternatives = audit_probs_bias or {}
            audit_values: Dict[str, float] = {}
            for proposal_id in self.audit_proposal_ids:
                if proposal_id not in alternatives:
                    raise SimulationConfigurationError(
                        f"Missing categorical probabilities for audit proposal {proposal_id!r}."
                    )
                probs_alt = _normalize_probs(alternatives[proposal_id])
                if len(probs_alt) != len(probs_a):
                    raise ValueError(
                        "Audit and nominal categorical probabilities must have the same length."
                    )
                if any(
                    p_base > 0.0 and p_alt == 0.0
                    for p_base, p_alt in zip(probs_a, probs_alt)
                ):
                    raise ValueError(
                        f"Audit proposal {proposal_id!r} excludes a base-supported outcome."
                    )
                log_p_alt = -math.inf if probs_alt[idx] == 0.0 else math.log(probs_alt[idx])
                audit_values[proposal_id] = log_p_base - log_p_alt
            self._record_audit_log_ratios(audit_values, component)
        return idx

    def normal(
        self, mu_base: float, sigma_base: float, mu_bias: float, sigma_bias: float,
        *, component: str = "unlabeled",
        audit_params_bias: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> float:
        """
        Draw from N(mu_bias, sigma_bias) and accumulate the likelihood ratio
        against N(mu_base, sigma_base).
        """
        x, replayed = self._sample_or_replay(
            "normal", component,
            lambda: self.rng.gauss(mu_base, sigma_base)
            if self._sample_base_component()
            else self.rng.gauss(mu_bias, sigma_bias),
        )
        if replayed and (float(mu_base), float(sigma_base)) != (
            float(mu_bias), float(sigma_bias)
        ):
            raise SimulationConfigurationError(
                "Stochastic-prefix replay is allowed only under the nominal law."
            )
        x = float(x)
        log_p_base = _logpdf_normal(x, mu_base, sigma_base)
        log_p_bias = _logpdf_normal(x, mu_bias, sigma_bias)
        self._record_log_ratio(
            log_p_base - log_p_bias, component, kind="normal",
            sampled_value=x, replayed=replayed,
        )
        if self.audit_proposal_ids:
            alternatives = audit_params_bias or {}
            audit_values = {}
            for proposal_id in self.audit_proposal_ids:
                params = alternatives.get(proposal_id, (mu_base, sigma_base))
                if len(params) != 2:
                    raise ValueError("Audit normal parameters must be (mu, sigma).")
                log_p_alt = _logpdf_normal(x, float(params[0]), float(params[1]))
                audit_values[proposal_id] = log_p_base - log_p_alt
            self._record_audit_log_ratios(audit_values, component)
        return x

    def lognormal(
        self, mu_base: float, sigma_base: float, mu_bias: float, sigma_bias: float,
        *, component: str = "unlabeled",
    ) -> float:
        """
        Draw from LogNormal(mu_bias, sigma_bias) and accumulate the likelihood
        ratio against LogNormal(mu_base, sigma_base).
        """
        x, replayed = self._sample_or_replay(
            "lognormal", component,
            lambda: self.rng.lognormvariate(mu_base, sigma_base)
            if self._sample_base_component()
            else self.rng.lognormvariate(mu_bias, sigma_bias),
        )
        if replayed and (float(mu_base), float(sigma_base)) != (
            float(mu_bias), float(sigma_bias)
        ):
            raise SimulationConfigurationError(
                "Stochastic-prefix replay is allowed only under the nominal law."
            )
        x = float(x)
        log_p_base = _logpdf_lognormal(x, mu_base, sigma_base)
        log_p_bias = _logpdf_lognormal(x, mu_bias, sigma_bias)
        self._record_log_ratio(
            log_p_base - log_p_bias, component, kind="lognormal",
            sampled_value=x, replayed=replayed,
        )
        return x

    def shifted_lognormal(
        self,
        shift_base: float,
        mu_base: float,
        sigma_base: float,
        shift_bias: float,
        mu_bias: float,
        sigma_bias: float,
        *,
        component: str = "unlabeled",
        audit_params_bias: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> float:
        """Draw a shifted log-normal and weight the realized total value.

        The base and proposal shifts may differ.  Weighting only the unshifted
        proposal variate is incorrect in that case because the base density
        must be evaluated at ``total - shift_base``.
        """
        def fresh_total() -> float:
            if self._sample_base_component():
                variable_sample = self.rng.lognormvariate(mu_base, sigma_base)
                return float(shift_base) + variable_sample
            variable_sample = self.rng.lognormvariate(mu_bias, sigma_bias)
            return float(shift_bias) + variable_sample

        total, replayed = self._sample_or_replay(
            "shifted_lognormal", component, fresh_total,
        )
        if replayed and (
            float(shift_base), float(mu_base), float(sigma_base)
        ) != (
            float(shift_bias), float(mu_bias), float(sigma_bias)
        ):
            raise SimulationConfigurationError(
                "Stochastic-prefix replay is allowed only under the nominal law."
            )
        total = float(total)
        variable_base = total - float(shift_base)
        variable_bias = total - float(shift_bias)
        log_p_base = _logpdf_lognormal(variable_base, mu_base, sigma_base)
        log_p_bias = _logpdf_lognormal(variable_bias, mu_bias, sigma_bias)
        self._record_log_ratio(
            log_p_base - log_p_bias, component, kind="shifted_lognormal",
            sampled_value=total, replayed=replayed,
        )
        if self.audit_proposal_ids:
            alternatives = audit_params_bias or {}
            audit_values = {}
            for proposal_id in self.audit_proposal_ids:
                params = alternatives.get(
                    proposal_id, (shift_base, mu_base, sigma_base)
                )
                if len(params) != 3:
                    raise ValueError(
                        "Audit shifted-lognormal parameters must be (shift, mu, sigma)."
                    )
                shift_alt, mu_alt, sigma_alt = map(float, params)
                log_p_alt = _logpdf_lognormal(
                    total - shift_alt, mu_alt, sigma_alt
                )
                audit_values[proposal_id] = log_p_base - log_p_alt
            self._record_audit_log_ratios(audit_values, component)
        return total

    def uniform(
        self, a_base: float, b_base: float, a_bias: float, b_bias: float,
        *, component: str = "unlabeled",
    ) -> float:
        """
        Draw from Uniform[a_bias, b_bias] and accumulate the likelihood ratio
        against Uniform[a_base, b_base]. If the sampled point lies outside the
        base support, the log-weight is set to -inf.
        """
        if b_bias <= a_bias:
            raise ValueError("Bias bounds must satisfy b_bias > a_bias.")
        if b_base <= a_base:
            raise ValueError("Base bounds must satisfy b_base > a_base.")

        x, replayed = self._sample_or_replay(
            "uniform", component,
            lambda: self.rng.uniform(a_base, b_base)
            if self._sample_base_component()
            else self.rng.uniform(a_bias, b_bias),
        )
        if replayed and (float(a_base), float(b_base)) != (
            float(a_bias), float(b_bias)
        ):
            raise SimulationConfigurationError(
                "Stochastic-prefix replay is allowed only under the nominal law."
            )
        x = float(x)
        if x < a_base or x > b_base:
            self._record_log_ratio(
                -math.inf, component, kind="uniform",
                sampled_value=x, replayed=replayed,
            )
            return x

        width_base = b_base - a_base
        width_bias = b_bias - a_bias
        log_p_base = math.log(1.0 / width_base)
        log_p_bias = -math.inf if x < a_bias or x > b_bias else math.log(1.0 / width_bias)
        self._record_log_ratio(
            log_p_base - log_p_bias, component, kind="uniform",
            sampled_value=x, replayed=replayed,
        )
        return x

    # --------- manual hook ---------

    def incorporate_logpdf(
        self, log_p_base: float, log_p_bias: float, *, component: str = "unlabeled"
    ) -> None:
        """
        Manually inject a likelihood ratio when a draw is handled externally.
        """
        self._record_log_ratio(
            log_p_base - log_p_bias, component, kind="external",
            sampled_value=None, replayed=False,
        )
