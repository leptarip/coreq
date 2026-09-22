#!/usr/bin/env python
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
import os
import time
import timeit
import random

import carla
import numpy as np

import source.simulation_environment.utils as utils
import source.simulation_environment.logs.printl
import source.simulation_environment.cfg.cfg_parser as cfg_parser
import source.simulation_environment.scenario.scenarios as scenarios
import source.simulation_environment.logs.redis_data as rd
from source.simulation_environment.logs.printl import PrintL
from source.simulation_environment.logs.redis_streams import RedisSimulationStream
from source.simulation_environment.scenario.base_scenario import BaseScenario
import source.simulation_environment.sim_utils as sim_utils
from source.simulation_environment.configuration_error import (
    SimulationConfigurationError,
    require_valid_configuration,
)
from source.simulation_environment.rare_events.wrapper import (
    ImportanceSampler,
    replay_configuration_is_nominal,
)
from source.simulation_environment.rare_events.seeds import (
    SAMPLER_SEED_ALGORITHM,
    derive_episode_sampler_seeds,
)

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*invalid value encountered in line_locate_point.*",
    category=RuntimeWarning,
    module=r"shapely\.linear",
)


def save_data(episode_num: int, exp_name: str, root_folder: str, printl: source.simulation_environment.logs.printl.PrintL,
              rare_event_data=None, execution_data=None):
    sim_data = RedisSimulationStream()
    file_name = root_folder + "/" + exp_name + "/sim_data_" + str(episode_num) + ".json"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    sim_data.get_data(exp_name)
    data_out = sim_data.write_to_file(
        file_name,
        rare_event_data=rare_event_data,
        execution_data=execution_data,
    )
    if isinstance(data_out, dict):
        execution = dict(data_out.get("execution", {}) or {})
        execution["simulator_trajectory_path"] = os.path.abspath(file_name)
        data_out["execution"] = execution

    try:
        sim_data.delete_streams(exp_name)
    except Exception as e:
        print("exception in deleting streams {0}".format(e))

    file_name = root_folder + "/" + exp_name + "/log_" + str(episode_num) + ".txt"
    printl.write_to_file(file_name)

    # write entry completed to episode log
    try:
        file_name = root_folder + "/episodes_progress" + ".txt"
        content = "completed episode {0}{1}".format(episode_num, os.linesep)
        with open(file_name, 'a+') as f:
            f.write(content)
    except Exception as e:
        print("Exception on saving episodes_progress {0}".format(e))

    return data_out


def get_episode_num_from_progress(root_folder: str) -> int:
    episode = 0
    # read entry completed to episode log
    file_name = root_folder + "/episodes_progress" + ".txt"

    try:
        with open(file_name, 'r') as f:
            lines = list()
            for line in f:
                lines.append(line)
            if len(lines) > 0:
                nums = [int(word) for word in lines[-1].split() if word.isdigit()]
                episode = nums[-1] + 1
    except Exception as e:
        print("exception in deleting streams {0}".format(e))

    return episode

def apply_sim_settings(settings, cfg: dict):
    """Populate CARLA world settings from the sim config.

    Factored out of ``main`` so the config-to-settings mapping can be pinned by a
    unit test without a live server. The same settings object is re-applied after
    every ``reload_world(reset_settings=False)``, so anything set here persists for
    the whole job.

    ``no_rendering_mode`` defaults to False: nothing in this simulator consumes a
    rendered frame (perception is the geometric FOV model in ``obps/``, and the only
    camera blueprint in the tree belongs to the debug window, which the runner never
    constructs), but leaving the default unchanged keeps existing campaign configs
    byte-identical until they opt in.

    :param settings: a carla.WorldSettings from ``world.get_settings()``
    :param cfg: parsed simulator config
    :return: the same settings object, populated
    """
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg["sim"]["delta_time"]
    # physics sub stepping
    settings.substepping = True
    settings.no_rendering_mode = bool(cfg["sim"].get("no_rendering_mode", False))
    return settings


def main(folder_root: str, cfg_path: str):
    printl = source.simulation_environment.logs.printl.PrintL("Runner", enabled=True)
    # load configuration
    cfg = cfg_parser.parse(cfg_path)
    ret_cfg, desc_cfg = cfg_parser.validate_cfg(cfg)
    ret_cfg_rare_events, desc_cfg_rare_events = cfg_parser.validate_rare_event_cfg(cfg)
    try:
        require_valid_configuration(ret_cfg, desc_cfg, scope="simulator")
        require_valid_configuration(
            ret_cfg_rare_events,
            desc_cfg_rare_events,
            scope="rare-event",
        )
    except SimulationConfigurationError as exc:
        printl.to_print(message=str(exc), where=printl.STD_OUT_AND_MEM)
        raise

    total_episodes = int(cfg["sim"]["num_episodes"])
    base_seed = int(cfg.get("sim", {}).get("seed", time.time_ns() & 0xFFFFFFFF))
    seed_list_raw = cfg.get("sim", {}).get("seed_list")
    if seed_list_raw is not None:
        if not isinstance(seed_list_raw, list):
            raise SimulationConfigurationError(
                "sim.seed_list must be a list of integers when provided"
            )
        simulation_seeds = [int(seed) for seed in seed_list_raw]
        if len(simulation_seeds) != total_episodes:
            raise SimulationConfigurationError(
                f"sim.seed_list length {len(simulation_seeds)} does not match "
                f"sim.num_episodes {total_episodes}"
            )
    else:
        simulation_seeds = [base_seed + index for index in range(total_episodes)]

    re_cfg = cfg.get("rare_event", {})
    replay_map = re_cfg.get("splitting_replay_prefix_by_simulation_seed", {})
    has_replay = any(bool(prefix) for prefix in replay_map.values())
    if has_replay:
        if not re_cfg.get("splitting_replay_nominal_law", False):
            raise SimulationConfigurationError(
                "Stochastic-prefix replay requires a nominal-law declaration."
            )
        if not replay_configuration_is_nominal(cfg, re_cfg):
            raise SimulationConfigurationError(
                "splitting_replay_nominal_law conflicts with biased simulator parameters."
            )
    sampler_seeds = None
    if re_cfg.get("enabled", False):
        sampler_seeds = derive_episode_sampler_seeds(
            simulation_seeds,
            re_cfg.get("seed"),
        )

    # set-up redis keys
    while 1:
        h_set = utils.create_experiment_name()
        res = rd.redis_instance.get(h_set)
        if res is None:
            break

    rd.set_exp_name(h_set)  # only for visualization

    # Connect to carla server
    client = carla.Client(cfg["server_ip"], cfg["server_port"])
    client.set_timeout(10.0)

    # Loading the world
    world = client.load_world(scenarios.get_scenario_map(cfg))
    # Enables synchronous mode
    settings = apply_sim_settings(world.get_settings(), cfg)
    printl.to_print(message="max_substep_delta_time: {0}".format(settings.max_substep_delta_time))
    if not settings.fixed_delta_seconds <= settings.max_substep_delta_time * settings.max_substeps:
        message = (
            "sim.delta_time exceeds CARLA's configured physics-substep capacity: "
            f"{settings.fixed_delta_seconds} > "
            f"{settings.max_substep_delta_time * settings.max_substeps}"
        )
        printl.to_print(message=message, where=printl.STD_OUT_AND_MEM)
        raise SimulationConfigurationError(message)

    world.apply_settings(settings)

    starting_episode = get_episode_num_from_progress(folder_root)

    sim_time_step = sim_utils.get_sim_step(settings.fixed_delta_seconds)
    data_out = list()
    sampler = None
    if re_cfg.get("enabled", False):
        sampler = ImportanceSampler(
            defensive_mixture_probability=float(
                re_cfg.get("defensive_mixture_probability", 0.0)
            ),
            record_draw_history=bool(re_cfg.get("splitting_record_draw_history", False)),
            audit_proposal_ids=tuple(
                str(value) for value in (
                    re_cfg.get("audit_proposal_ids")
                    or re_cfg.get("sensor_bias", {}).get("audit_miss_scales", {})
                    or {}
                )
            ),
        )

    # iterate for the number of runs
    for i in range(starting_episode, total_episodes):
        # set-up redis keys
        rd.set_up_new_keys(h_set, i)

        ep_seed = simulation_seeds[i]
        root_seed_map = re_cfg.get("splitting_root_seed_by_simulation_seed", {})
        root_seed = int(root_seed_map.get(str(ep_seed), ep_seed))
        random.seed(root_seed)
        np.random.seed(root_seed)

        sampler_seed = None
        if sampler:
            sampler_seed = sampler_seeds[i]
            replay_prefix = replay_map.get(str(ep_seed), [])
            sampler.reset(
                seed=sampler_seed,
                replay_prefix=replay_prefix,
                replay_under_nominal_law=bool(
                    re_cfg.get("splitting_replay_nominal_law", False)
                ),
            )

        world = client.reload_world(reset_settings=False)
        world.apply_settings(settings)
        carla_map = world.get_map()

        # get scenario
        scenario = scenarios.get_scenario(experiment_name=h_set,
                                          cfg=cfg,
                                          world=world,
                                          carla_map=carla_map,
                                          delta_time=settings.fixed_delta_seconds,
                                          sampler=sampler,
                                          seed=root_seed)
        try:
            # Generate and prepare inside the cleanup boundary: both operations
            # may fail after actors have already been spawned.
            result = scenario.generate_scenario()
            if result != BaseScenario.ErrorCode.OK:
                message = f"Scenario configuration rejected during construction ({result.name})."
                printl.to_print(message=message, force=True, where=PrintL.STD_OUT_AND_MEM)
                raise SimulationConfigurationError(message)
            extra = scenario.pre_run()
        except BaseException:
            try:
                scenario.destroy()
            except Exception as cleanup_error:
                printl.to_print(
                    message=f"Error cleaning up partially constructed scenario: {cleanup_error}",
                    where=PrintL.STD_OUT_AND_MEM,
                )
            finally:
                rd.close_instance()
            raise

        try:
            sim_timer = timeit.default_timer()
            printl.to_print(message=f'--Episode {i + 1} of {int(cfg["sim"]["num_episodes"])}--',force=True)

            printl.to_print(
                message=f"--Simulation Starts, fixed_delta_seconds: {settings.fixed_delta_seconds}--",force=True)

            sim_time, end_cause = run_sim(sim_time_step, world, scenario, printl, extra)

            printl.to_print(message="--total time : {0}, fixed_delta_seconds: {1}"
                            .format(timeit.default_timer() - sim_timer, settings.fixed_delta_seconds),
                            force=True)

            printl.to_print(message="--saving data", force=True)
            rare_event_payload = None
            if sampler and re_cfg.get("enabled", False):
                if not sampler.replay_prefix_consumed:
                    raise SimulationConfigurationError(
                        "The episode ended before consuming its complete splitting replay prefix."
                    )
                rare_event_payload = {
                    "enabled": True,
                    "seed": sampler_seed,
                    "seed_algorithm": SAMPLER_SEED_ALGORITHM,
                    "simulation_seed": ep_seed,
                    "log_weight": sampler.log_weight,
                    "weight": sampler.weight(),
                    "component_log_ratio": sampler.component_log_ratio,
                    "component_log_ratios": sampler.component_log_ratios,
                    "component_draw_counts": sampler.component_draw_counts,
                    "audit_proposal_log_ratios": sampler.audit_proposal_log_ratios,
                    "audit_proposal_component_log_ratios": (
                        sampler.audit_proposal_component_log_ratios
                    ),
                    "proposal_component": sampler.proposal_component,
                    "defensive_mixture_probability": sampler.defensive_mixture_probability,
                    "draw_count": sampler.draw_count,
                    "root_simulation_seed": root_seed,
                    "bias_cfg": {
                        key: value for key, value in re_cfg.items()
                        if key not in {
                            "splitting_replay_prefix_by_simulation_seed",
                            "splitting_root_seed_by_simulation_seed",
                            "splitting_replay_nominal_law",
                        }
                    }
                }
                if sampler.record_draw_history:
                    rare_event_payload.update({
                        "draw_history": sampler.draw_history,
                        "replay_prefix_length": sampler.replay_prefix_length,
                        "replay_prefix_consumed": sampler.replay_prefix_consumed,
                    })
            execution_payload = {
                "simulation_seed": int(ep_seed),
                "root_simulation_seed": int(root_seed),
                "episode_index": int(i),
            }
            sim_result = save_data(
                i,
                h_set,
                folder_root,
                printl,
                rare_event_data=rare_event_payload,
                execution_data=execution_payload,
            )
            data_out.append(sim_result)

            printl.to_print(message="---------END---------", force=True)

            #print("##sub_dir {0}, episode {1} of {2}##".format(folder_root, i + 1, total_episodes))
            print(f"--episode: {h_set}:{i} total time : {timeit.default_timer() - sim_timer}, "
                  f"fixed_delta_seconds: {settings.fixed_delta_seconds} cause: {scenario.state_to_string(end_cause)}")

        finally:
            # Actor.destroy() is a blocking RPC and the next episode reloads the
            # world outright, so nothing here needs waiting on.
            scenario.destroy()
            rd.close_instance()

    return data_out


def run_sim(sim_step: int, world, scenario: BaseScenario, printl: PrintL, extra):
    """
        Simulation loop
    :param sim_step:
    :param world:
    :param scenario:
    :param printl:
    :param extra:
    :return:
    """
    sim_time = 0
    while 1:
        printl.to_print(message="---------New Tick sim_time: {0}-------------".format(sim_time), force=True)
        # Carla Tick - we are in sync mode
        world.tick(10000)
        sampler = getattr(scenario, "sampler", None)
        if sampler is not None:
            sampler.set_context_time_ms(sim_time)
        # termination condition
        sim_state = scenario.run_step(sim_time, extra)

        if sim_state != scenario.State.IN_PROGRESS:
            return sim_time, sim_state
        sim_time = sim_time + sim_step


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one simulation against a CARLA server on the config's server_port "
                    "(Redis must be running on localhost:6379)."
    )
    parser.add_argument("--config", required=True, help="Scenario TOML, e.g. configuration/config_intersection1_debug.toml")
    parser.add_argument("--out", default="output/simulation", help="Output folder for the simulation results.")
    args = parser.parse_args()
    main(args.out, args.config)
