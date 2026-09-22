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
# aoi_block_idx.py
from __future__ import annotations

from typing import Sequence, Union

import numpy as np

Number = Union[int, float]
Quantiles = Union[float, Sequence[float]]


def get_average_peaks(aoi_values, idx_start, idx_end):
    try:
        vals = aoi_values[idx_start:idx_end]
        peaks_times = []
        peak_vals = []
        for i in range(1, len(vals) - 1):
            if vals[i] > vals[i - 1] and vals[i] > vals[i + 1]:
                peaks_times.append(i)
                peak_vals.append(vals[i])

        vals = np.array(vals)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None

        avg = np.average(vals)

        return avg
    except Exception as e:
        print("[EXCEPTION] get_average_peaks: ", e)
        return None

def get_aoi_percentiles(aoi_values, idx_start, idx_end):
    try:
        vals = np.array(aoi_values[idx_start:idx_end])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return { "p90": None,
                     "p95": None,
                     "p99": None,
                     "p100": None
                     }
        p = [90, 95, 99, 100]

        p90, p95, p99, p100 = np.percentile(vals, p)

        return {
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "p100": p100
        }
    except ValueError as e:
        print("[EXCEPTION] get_aoi_percentiles: ", e)
        return {"p90": None,
                "p95": None,
                "p99": None,
                "p100": None
                }

def get_aoi_mean(aoi_values, idx_start, idx_end):
    try:
        vals = np.array(aoi_values[idx_start:idx_end])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None

        avg = np.average(vals)

        return avg
    except Exception as e:
        print("[EXCEPTION] get_aoi_mean: ", e)
        return None

def parse(parsed: dict):

    # parse pieces
    step = int(parsed["scenario"]["delta_t"] / 0.001)
    ego = parsed["ego"]
    adv = parsed["adv"]

    ego_tactical =      ego["tactical_data"]
    block_id = next(iter(ego["perception_data"]["blocks"]))
    perception_block =  ego["perception_data"]["blocks"][block_id]

    # get min dist and max vel at min dist
    dist_to_target = perception_block.get("dist_to_target", []) or []
    distance_times = perception_block.get("t", []) or []
    if len(distance_times) != len(dist_to_target):
        raise ValueError(
            "Perception distance timestamps and values must have equal length "
            f"(got {len(distance_times)} and {len(dist_to_target)})."
        )

    def valid_distance(value):
        return (
            value is not None
            and np.isfinite(float(value))
            and float(value) > -1.0
        )

    distance_trace = [
        {"time_ms": int(time_ms), "distance_m": float(distance_m)}
        for time_ms, distance_m in zip(distance_times, dist_to_target)
        if valid_distance(distance_m)
    ]
    valid_distance_indices = [
        index for index, value in enumerate(dist_to_target) if valid_distance(value)
    ]
    t_min_dist = None
    ego_vel_at_min_dist = None
    adv_vel_at_min_dist = None

    if valid_distance_indices:
        idx = min(valid_distance_indices, key=lambda index: float(dist_to_target[index]))
        min_dist_to_target = float(dist_to_target[idx])
        # perception block time t is subject to the ad period
        t_values = perception_block.get("t", []) or []
        if idx < len(t_values):
            t_min_dist = t_values[idx]
            idx_sim = int(t_min_dist / step) - 1
            # ego and adv log blocks are given per each simulator tick
            # we need to transform the index from perception block to the "global one"
            ego_vel = ego["data"].get("vel", []) if isinstance(ego.get("data"), dict) else []
            adv_vel = adv["data"].get("vel", []) if isinstance(adv.get("data"), dict) else []
            if 0 <= idx_sim < len(ego_vel):
                ego_vel_at_min_dist = ego_vel[idx_sim]
            if 0 <= idx_sim < len(adv_vel):
                adv_vel_at_min_dist = adv_vel[idx_sim]
    else:
        # No valid distance observations in this rollout; keep parser robust and explicit.
        min_dist_to_target = None

    aoi_perc_first_last_msg = None
    aoi_avg_first_last_msg = None
    aoi_peak_avg_first_last_msg = None
    aoi_perc_first_msg_min_dist = None
    aoi_avg_first_msg_min_dist = None
    aoi_peak_avg_first_msg_min_dist = None
    first_msg_t = None
    pred_target_ttcr = None
    gt_target_ttcr = None
    

    if len(ego_tactical["blocks"])>0:
        block_id =          next(iter(ego_tactical["blocks"]))
        block =             ego_tactical["blocks"][block_id]
        ego_info =          block["ego_info"]
        target_info =       block["target_info"]

        if len(block["t"]) > 0:
            first_msg_t = block["t"][0]

        # ----- Get AoI values in intervals

        # first to last msg
        if len(block["msg_id"]) > 0:
            aoi_perc_first_last_msg =       get_aoi_percentiles(block["aoi"], 0, len(block["msg_id"]))
            aoi_avg_first_last_msg =        get_aoi_mean(block["aoi"], 0, len(block["msg_id"]))
            aoi_peak_avg_first_last_msg =   get_average_peaks(block["aoi"], 0, len(block["msg_id"]))

        # first msg to min target distance
        try:
            idx_end = block["t"].index(t_min_dist)
            if len(block["msg_id"]) > 0:
                aoi_perc_first_msg_min_dist =       get_aoi_percentiles(block["aoi"], 0, idx_end + 1)
                aoi_avg_first_msg_min_dist =        get_aoi_mean(block["aoi"], 0, idx_end + 1)
                aoi_peak_avg_first_msg_min_dist =   get_average_peaks(block["aoi"], 0, idx_end + 1)
        except Exception as e:
            #the index cannot be found, there were no messages in tactical block
            pass        

    # ---- Get collision result
    collision = True if parsed["scenario"]["cause"] == "col_shapes" else False


    out_dict = {
         "col": collision,
         "first_msg_t": first_msg_t,
         "aoi_perc_first_last_msg":     aoi_perc_first_last_msg,
         "aoi_avg_first_last_msg":      aoi_avg_first_last_msg,
         "aoi_peak_avg_first_last_msg": aoi_peak_avg_first_last_msg,
         "aoi_perc_first_msg_min_dist": aoi_perc_first_msg_min_dist,
         "aoi_avg_first_msg_min_dist":  aoi_avg_first_msg_min_dist,
         "aoi_peak_avg_first_msg_min_dist": aoi_peak_avg_first_msg_min_dist,
         "min_dist_to_target":  min_dist_to_target,
         "min_dist_t_ms": t_min_dist,
         "distance_trace": distance_trace,
         "ego_vel_at_min_dist": ego_vel_at_min_dist,
         "adv_vel_at_min_dist": adv_vel_at_min_dist,
         "execution": parsed.get("execution", {}),
         "rare_event": parsed.get("rare_event", {})
    }

    return out_dict




