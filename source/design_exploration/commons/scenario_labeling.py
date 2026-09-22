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
from dataclasses import dataclass
from typing import Tuple, Callable, List, Optional
import math
import time
import logging
from enum import IntEnum
from copy import deepcopy
from scipy.stats import beta

from source.design_exploration.commons.episodes.episode_result import EpisodeResult
from source.design_exploration.commons.design_space import Design
from source.design_exploration.simulator_interface.protocols import SimulationInterface

logger = logging.getLogger(__name__)

# ---------- CP utilities ----------
def cp_upper_bound(k: int, n: int, delta: float) -> float:
    if n == 0: return 1.0
    if k >= n: return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))

def required_min_iters_k0(P: float, alpha: float) -> int:
    return int(math.ceil(math.log(alpha) / math.log(1.0 - P)))

# ---------- SPRT config & tester ----------
@dataclass
class SPRTConfig:
    P: float = 0.05
    C: float = 0.95
    p1_factor: float = 0.5
    beta_err: float = 0.2
    global_max_iters: int = 80
    batch_size: int = 10


WALD_SAFE = "WALD_SAFE"
WALD_UNSAFE = "WALD_UNSAFE"
CP_K0 = "CP_K0"
CP_CAP = "CP_CAP"
CAP_UNSAFE= "CAP_UNSAFE"
OBS_FAIL= "OBS_FAIL"

class TestResult(IntEnum):        
    SAFE = 1
    UNSAFE = 0


@dataclass(frozen=True)
class SPRTDecision:
    """A terminal decision produced by the shared online/offline state machine."""

    label: TestResult
    mode: str


def sprt_log_likelihood(n: int, k: int, cfg: SPRTConfig) -> float:
    """Return the Wald log likelihood ratio for ``n`` trials and ``k`` failures."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"Invalid SPRT counts n={n}, k={k}.")
    p0 = float(cfg.P)
    p1 = float(cfg.p1_factor) * p0
    return (
        k * math.log(p1 / p0)
        + (n - k) * math.log((1.0 - p1) / (1.0 - p0))
    )


def sprt_decision_for_counts(
    n: int, k: int, cfg: SPRTConfig
) -> Optional[SPRTDecision]:
    """Apply the production stopping rules to sufficient statistics."""
    log_l = sprt_log_likelihood(n, k, cfg)
    alpha_err = 1.0 - float(cfg.C)
    p0 = float(cfg.P)
    log_a = math.log((1.0 - float(cfg.beta_err)) / alpha_err)
    log_b = math.log(float(cfg.beta_err) / (1.0 - alpha_err))
    min_iters_k0 = required_min_iters_k0(p0, alpha_err)

    if k == 0 and n >= min_iters_k0:
        return SPRTDecision(TestResult.SAFE, CP_K0)
    if log_l >= log_a:
        return SPRTDecision(TestResult.SAFE, WALD_SAFE)
    if log_l <= log_b:
        return SPRTDecision(TestResult.UNSAFE, WALD_UNSAFE)
    if n >= int(cfg.global_max_iters):
        upper = cp_upper_bound(k, n, alpha_err)
        if upper <= p0:
            return SPRTDecision(TestResult.SAFE, CP_CAP)
        return SPRTDecision(TestResult.UNSAFE, CAP_UNSAFE)
    return None


@dataclass
class SPRTState:
    """Shared sequential state used by simulation and cached-episode replay."""

    cfg: SPRTConfig
    n: int = 0
    k: int = 0

    @property
    def log_likelihood(self) -> float:
        return sprt_log_likelihood(self.n, self.k, self.cfg)

    @property
    def decision(self) -> Optional[SPRTDecision]:
        return sprt_decision_for_counts(self.n, self.k, self.cfg)

    def observe(self, result: TestResult) -> Optional[SPRTDecision]:
        if self.decision is not None:
            raise RuntimeError("Cannot append an observation after an SPRT decision.")
        self.n += 1
        if result == TestResult.UNSAFE:
            self.k += 1
        elif result != TestResult.SAFE:
            raise ValueError(f"Unknown test result {result!r}.")
        return self.decision

class TestInfo:
    def __init__(self, n:int, k:int, mode:str, raw):
        self.n=n
        self.k=k
        self.mode=mode
        self.raw=raw
        

class ScenarioTester:
    """Sequential per-design tester driven by a user-supplied `violation_function`.
    """
    def __init__(self, 
                 sim_iface: SimulationInterface,
                 scenario_name: str, 
                 scenario_base_cfg: dict, 
                 cfg: SPRTConfig,
                 violation_function: Callable[[EpisodeResult], TestResult]):
        
        self.sim = sim_iface
        self.scenario_name = scenario_name
        # Keep a copy of the SPRT settings so offline warm-start can reuse them.
        self.sprt_cfg = deepcopy(cfg)
        self.P = cfg.P
        self.alpha_err = 1.0 - cfg.C
        self.beta_err = cfg.beta_err
        self.p0 = cfg.P
        self.p1 = cfg.p1_factor * cfg.P
        self.global_max_iters = cfg.global_max_iters
        self.batch_size = cfg.batch_size
        self.logA = math.log((1.0 - self.beta_err) / self.alpha_err)
        self.logB = math.log(self.beta_err / (1.0 - self.alpha_err))
        self.min_iters_k0 = required_min_iters_k0(self.P, self.alpha_err)
        self.base_cfg = scenario_base_cfg
        self.violation_function = violation_function
        self._max_restarts = 5
   
    def _episodes(self, design: Design, num_eps: int) -> List[EpisodeResult]:
        """Run exactly n fresh episodes; auto-restart sim on failure."""
        attempts = 0
        while True:
            try:
                return self.sim.simulate(self.scenario_name, self.base_cfg, design, num_eps)
            except Exception as e:
                if attempts >= self._max_restarts:
                    raise
                self.restart_sim(e)
                attempts += 1

    def sprt_label(self, design: Design) -> Tuple[TestResult, TestInfo]:
        state = SPRTState(self.sprt_cfg)
        raw = []
        while True:
            batch = self._episodes(design, self.batch_size)
            raw.extend(batch)
            for out in batch:
                vio = self.violation_function(out)
                decision = state.observe(vio)
                if decision is not None:
                    return decision.label, TestInfo(
                        n=state.n, k=state.k, mode=decision.mode, raw=raw
                    )

    def cp_only_label(self, design: Design) -> Tuple[TestResult, TestInfo]:
        # Reaching global_max_iters without early UNSAFE implies k == 0
        # (we return immediately on first UNSAFE in OBS_FAIL).
        n = 0; k = 0; raw = []
        while True:
            batch = self._episodes(design, self.batch_size)
            raw.extend(batch)
            for out in batch:
                vio = self.violation_function(out)
                n += 1
                if vio == TestResult.UNSAFE:
                    k += 1
                    return TestResult.UNSAFE, TestInfo(n=n, k=k, mode=OBS_FAIL, raw=raw)
                if n >= self.min_iters_k0:
                    return TestResult.SAFE, TestInfo(n=n, k=0, mode=CP_K0, raw=raw)
                if n >= self.global_max_iters:
                    ub = cp_upper_bound(k, n, self.alpha_err)
                    if ub <= self.P:
                        return TestResult.SAFE, TestInfo(n=n, k=0, mode=CP_CAP, raw=raw)
                    else:
                        return TestResult.UNSAFE, TestInfo(n=n, k=k, mode=CAP_UNSAFE, raw=raw)
                    
    def restart_sim(self, e):
        logger.exception("Launching Carla instance exception: %s", e)
        logger.info("RESTARTING SIM_INTERFACE")
        try:
            self.sim.close()
        except Exception:
            pass
        time.sleep(10)
        n_inst = getattr(self.sim, "num_carla_instances", 3)
        sim_cls = self.sim.__class__
        kwargs = dict(
            output_folder=self.sim.output_folder,
            algo_version=self.sim.algo_version,
            num_carla_instances=n_inst,
        )
        carla_path = getattr(self.sim, "carla_path", None)
        if carla_path is not None:
            kwargs["carla_path"] = carla_path
        try:
            self.sim = sim_cls(**kwargs)
        except TypeError:
            kwargs.pop("carla_path", None)
            self.sim = sim_cls(**kwargs)
