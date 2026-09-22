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

from source.simulation_environment.communication.fault_models.sit import SITLogic
SIT_LOGIC = "sit"

ALL_LOGIC = [SIT_LOGIC]

def get_fault_model(delta_time: float, cfg:dict, sampler=None, bias_cfg=None):
    """

    :param delta_time:
    :param cfg: it is the "fault" block
    :return:
    """
    if cfg["type"] == SIT_LOGIC:
        return SITLogic(delta_time, cfg, sampler=sampler, bias_cfg=bias_cfg)

    return None

def validate_cfg(cfg:dict) -> Tuple[bool, str]:

    if cfg["fault"]["type"] not in ALL_LOGIC:
        return False, "Unknown fault type"

    if cfg["fault"]["type"] == SIT_LOGIC:
        return SITLogic.validate_cfg(cfg["fault"])

    return False, "Problem validating fault type"


def validate_rare_event_cfg(re_cfg: dict, comm_cfg: Optional[dict] = None) -> Tuple[bool, str]:
    """
    Validate optional rare_event.fault_bias section.
    """
    fault_bias = re_cfg.get("fault_bias", {})
    if not fault_bias:
        return True, ""

    cfg_fault = comm_cfg["fault"] if comm_cfg and "fault" in comm_cfg else None
    if cfg_fault and cfg_fault.get("type") == SIT_LOGIC:
        supported = {
            "trigger_mean",
            "trigger_var",
            "trigger_mean_add",
            "trigger_var_scale",
            "audit_proposals",
        }
        unknown = sorted(set(fault_bias) - supported)
        if unknown:
            return False, f"Unsupported rare_event.fault_bias fields: {unknown}"
        variance_scale = fault_bias.get("trigger_var_scale")
        if variance_scale is not None and variance_scale <= 0:
            return False, "rare_event.fault_bias.trigger_var_scale must be > 0"
        variance = fault_bias.get("trigger_var")
        if variance is not None and variance <= 0:
            return False, "rare_event.fault_bias.trigger_var must be > 0"
        audit_proposals = fault_bias.get("audit_proposals", {}) or {}
        if not isinstance(audit_proposals, dict):
            return False, "rare_event.fault_bias.audit_proposals must be a mapping"
        for proposal_id, proposal in audit_proposals.items():
            if not str(proposal_id).strip() or not isinstance(proposal, dict):
                return False, "fault audit proposals require non-empty IDs and mappings"
            unknown_audit = sorted(set(proposal) - (supported - {"audit_proposals"}))
            if unknown_audit:
                return False, (
                    f"Unsupported fault audit fields for {proposal_id}: {unknown_audit}"
                )
            if float(proposal.get("trigger_var_scale", 1.0)) <= 0.0:
                return False, "fault audit trigger_var_scale must be > 0"

    return True, ""
