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
from typing import Tuple, Optional

from source.simulation_environment.communication.network_models.network_uniform import NetworkLogicUniform
from source.simulation_environment.communication.network_models.network_log_norm import NetworkLogicLogNorm

NETWORK_UNIFORM = "uniform"
NETWORK_LOG_NORM = "log_norm"

ALL_MODELS = [NETWORK_UNIFORM, NETWORK_LOG_NORM]

def get_network_model(cfg:dict, sampler=None, bias_cfg=None):
    """

    :param cfg: it is the "network" block
    :return:
    """
    if cfg["type"] == NETWORK_UNIFORM:
        return NetworkLogicUniform(cfg, sampler=sampler, bias_cfg=bias_cfg)
    elif cfg["type"] == NETWORK_LOG_NORM:
        return NetworkLogicLogNorm(cfg, sampler=sampler, bias_cfg=bias_cfg)

    return None


def validate_cfg(cfg:dict) -> Tuple[bool, str]:

    if cfg["network"]["type"] not in ALL_MODELS:
        return False, "Unknown network type"

    if cfg["network"]["type"] == NETWORK_UNIFORM:
        return NetworkLogicUniform.validate_cfg(cfg["network"])

    if cfg["network"]["type"] == NETWORK_LOG_NORM:
        return NetworkLogicLogNorm.validate_cfg(cfg["network"])

    return False, "Error in validating network model"


def validate_rare_event_cfg(re_cfg: dict, comm_cfg: Optional[dict] = None) -> Tuple[bool, str]:
    """
    Validate optional rare_event.network_bias section.
    """
    net_bias = re_cfg.get("network_bias", {})
    if not net_bias:
        return True, ""

    from source.simulation_environment.rare_events.window import validate_bias_window
    valid, reason = validate_bias_window(net_bias, "rare_event.network_bias")
    if not valid:
        return valid, reason

    network_cfg = comm_cfg.get("network") if comm_cfg else None
    if not network_cfg:
        return True, ""

    if network_cfg.get("type") == NETWORK_LOG_NORM:
        dm = net_bias.get("delay_min")
        da = net_bias.get("delay_avg")
        jit = net_bias.get("jitter")
        drop = net_bias.get("packet_drop_rate")

        if dm is not None and dm < 0:
            return False, "rare_event.network_bias.delay_min must be >= 0"
        if da is not None and da < 0:
            return False, "rare_event.network_bias.delay_avg must be >= 0"
        if dm is not None and da is not None and da < dm:
            return False, "rare_event.network_bias.delay_avg must be >= delay_min"
        if jit is not None and jit < 0:
            return False, "rare_event.network_bias.jitter must be >= 0"
        if drop is not None and not (0 <= drop <= 1):
            return False, "rare_event.network_bias.packet_drop_rate must be in [0,1]"
        
        return True, ""
        
    if network_cfg.get("type") == NETWORK_UNIFORM:
        d = net_bias.get("delay")
        jit = net_bias.get("jitter")
        drop = net_bias.get("packet_drop_rate")

        if d is not None and d < 0:
            return False, "rare_event.network_bias.delay must be >= 0"
        if jit is not None and jit < 0:
            return False, "rare_event.network_bias.jitter must be >= 0"
        if drop is not None and not (0 <= drop <= 1):
            return False, "rare_event.network_bias.packet_drop_rate must be in [0,1]"
        
        return True, ""

    return False, "Network type not supported for rare event validation"
