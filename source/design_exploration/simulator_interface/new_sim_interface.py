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

# sim_interface.py (stateless, no token replay; persists Parquet)
import multiprocessing
import os
import signal
import time
import logging
from datetime import datetime
import copy
import math
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock
import threading
from typing import List

import toml
from toml.encoder import TomlPreserveInlineDictEncoder

from source.design_exploration.commons.episodes.episode_result import EpisodeResult
import source.simulation_environment.runner.runner as runner
import source.simulation_environment.parsing_output.output_parser as output_parser
from source.design_exploration.commons.design_space import Design

logger = logging.getLogger(__name__)

_SUPPRESSED_CARLA_STDERR_MESSAGES = (
    "cannot parse georeference",
)


def _should_suppress_carla_stderr(text: str) -> bool:
    text = str(text)
    return any(msg in text for msg in _SUPPRESSED_CARLA_STDERR_MESSAGES)


def _drain_carla_stderr(stream, port: int) -> None:
    if stream is None:
        return
    try:
        for raw in iter(stream.readline, b""):
            if isinstance(raw, bytes):
                text = raw.decode("utf-8", errors="replace").rstrip()
            else:
                text = str(raw).rstrip()
            if not text or _should_suppress_carla_stderr(text):
                continue
            logger.warning("[carla:%d] %s", int(port), text)
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _start_carla_stderr_drain(proc: subprocess.Popen, port: int) -> None:
    if proc.stderr is None:
        return
    thread = threading.Thread(
        target=_drain_carla_stderr,
        args=(proc.stderr, int(port)),
        name=f"carla-stderr-{int(port)}",
        daemon=True,
    )
    thread.start()


class SimInterface:
    SERVER_IP = '127.0.0.1'
    BASE_PORT = 6000
    SUB_PROC_ERR = -1
    SUB_PROC_OK = 0
    CARLA_PATH = os.environ.get("CARLA_PATH", "")

    class CarlaServerDead(RuntimeError):
        def __init__(self, port: int, msg: str = "CARLA server died"):
            super().__init__(msg)
            self.port = port

    def __init__(self,
                 output_folder: str,
                 algo_version,
                 carla_path=CARLA_PATH,
                 num_carla_instances=3):
        self.carla_path = carla_path
        self.carla_instances = list()
        self.output_folder = output_folder
        self.algo_version = algo_version
        self.num_carla_instances = num_carla_instances
        self._counter_lock = Lock()
        self._executor = ProcessPoolExecutor(max_workers=num_carla_instances,
                                             mp_context=multiprocessing.get_context("spawn"))
        self._futures = []
        self.global_counter = 0

        # create output root folder
        now = datetime.now()
        timestamp = now.strftime("%d%m%y_%H%M%S") + f"_{now.microsecond // 1000:04d}"
        self.output_folder_path = os.path.join(output_folder, timestamp)
        os.makedirs(self.output_folder_path, exist_ok=True)

        # start carla instances
        for i in range(num_carla_instances):
            port = self.BASE_PORT + i * 1000
            dir_name, p_carla = self._start_carla_instance(port)
            if dir_name is not None and p_carla is not None:
                self.carla_instances.append({"num": i, "port": port, "subproc": p_carla, "in_use": False, "dir": dir_name})

    def _next_counter(self):
        with self._counter_lock:
            self.global_counter += 1
            return self.global_counter
        
    def _terminate_carla_instance(self, idx):
        entry = self.carla_instances[idx]
        entry["in_use"] = False
        p = entry.get("subproc")
        if not p or p.poll() is not None:
            entry["subproc"] = None
            return
        port = entry.get('port', 'N/A')
        logger.info("Terminating CARLA on port %s (parent PID: %s)", port, p.pid)
        try:
            pgid = os.getpgid(p.pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(3)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            logger.info("Process group for CARLA on port %s was already terminated.", port)
        except Exception as e:
            logger.error("Unexpected error during termination on port %s: %s", port, e)
        p.wait(timeout=5)
        entry["subproc"] = None
        logger.info("Termination complete for CARLA on port %s", port)

    def _start_carla_instance(self, port):
        err, p_carla = self._start_carla(port)
        retry = 0
        while err == SimInterface.SUB_PROC_ERR:
            logger.error("Failed to start carla sub process at port %d", port)
            if retry < 2:
                logger.info("Retrying...%d", port)
                err, p_carla = self._start_carla(port)
            else:
                return None, None
            retry += 1
        dir_name = os.path.join(self.output_folder_path, str(port))
        os.makedirs(dir_name, exist_ok=True)
        return dir_name, p_carla
    
    def _restart_carla_instance(self, idx: int):
        """
        Terminates the CARLA process at ``self.carla_instances[idx]`` and starts
        a fresh one on the **same port**.  The dictionary entry is updated in‑place.
        """
        instance = self.carla_instances[idx]
        port = instance["port"]
        logger.warning("Restarting CARLA on port %d", port)

        self._terminate_carla_instance(idx)

        # Relaunch the instance using the method that handles retries and dir creation
        dir_name, p_carla = self._start_carla_instance(port)
        if dir_name is None or p_carla is None:
            raise RuntimeError(f"Unable to restart CARLA on port {port} after multiple retries.")

        # Update the instance dictionary in-place
        instance["subproc"] = p_carla
        instance["dir"] = dir_name
        instance["in_use"] = False
        logger.info("Successfully restarted CARLA on port %d", port)
    
    def _start_carla(self, port):
        p_carla = None
        try:
            logger.info('launching carla on PORT: %d', port)
            carla_port = f'-carla-port={port}'
            streaming_port = f"-carla-streaming-port={port + 1}"
            p_carla = subprocess.Popen(
                [self.carla_path, "-RenderOffScreen", "-nosound", carla_port, streaming_port],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, preexec_fn=os.setsid)
            time.sleep(3)
            ret_code = p_carla.poll()
            if ret_code is not None:
                logger.error(p_carla.stderr.readline())
                return SimInterface.SUB_PROC_ERR, None
            logger.info('Launched CARLA return code %s', ret_code)
            _start_carla_stderr_drain(p_carla, port)
        except Exception as e:
            logger.exception("Launching Carla instance exception: %s", e)
            return SimInterface.SUB_PROC_ERR, p_carla
        return SimInterface.SUB_PROC_OK, p_carla

    def close(self):
        logger.info("Closing SimInterface: shutting down executor and terminating instances.")
        for fut in self._futures:
            fut.cancel()
        self._executor.shutdown(wait=True)
        self._futures.clear()
        for i in range(len(self.carla_instances)):
            self._terminate_carla_instance(i)

    def simulate(self, scenario_name: str, base_cfg : dict, design:Design, num_episodes: int)->List[EpisodeResult]:
        """Run fresh episodes
        Returns: list[dict] (parsed outputs for each episode)
        """
        try:            
            if num_episodes > len(self.carla_instances):
                sim_result = self._simulate_batch(scenario_name, base_cfg, design, num_episodes)
            else:
                try:
                    counter = self._next_counter()
                    sim_result = _simulate_single(base_cfg, self.SERVER_IP,
                                                self.carla_instances[0]["port"],
                                                self.carla_instances[0]["dir"],
                                                counter, 
                                                scenario_name,
                                                design, 
                                                num_episodes,
                                                self.algo_version)
                except SimInterface.CarlaServerDead as dead:
                    idx = next(i for i, d in enumerate(self.carla_instances) if d["port"] == dead.port)
                    self._restart_carla_instance(idx)
                    logger.info("Retrying the failed job on freshly restarted CARLA port %d", dead.port)
                    counter = self._next_counter()
                    sim_result = _simulate_single(base_cfg, self.SERVER_IP,
                                                self.carla_instances[idx]["port"],
                                                self.carla_instances[idx]["dir"],
                                                counter, 
                                                scenario_name, design, num_episodes,
                                                self.algo_version)

            return sim_result

        except Exception:
            logger.exception("simulate() failed; closing interface.")
            self.close()
            raise


    def _simulate_batch(self, scenario_name, base_cfg:dict, design:Design, num_episodes: int)->List[EpisodeResult]:
        remainder = num_episodes % len(self.carla_instances)
        base = num_episodes // len(self.carla_instances)
        episodes = [base + 1 if i < remainder else base for i in range(len(self.carla_instances))]

        sim_single_params = []
        for i, num_episodes in enumerate(episodes):
            counter = self._next_counter()
            sim_single_params.append((base_cfg, self.SERVER_IP,
                                      self.carla_instances[i]["port"], self.carla_instances[i]["dir"],
                                      counter, scenario_name, design, num_episodes, self.algo_version))

        results_all = []
        future_to_idx = {}
        for i, arg_tuple in enumerate(sim_single_params):
            future = self._executor.submit(_simulate_single, *arg_tuple)
            self._futures.append(future)
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results_all.extend(result)
            except SimInterface.CarlaServerDead:
                self._restart_carla_instance(idx)
                logger.info("Resubmitting batch job on restarted CARLA port %d", self.carla_instances[idx]["port"])
                arg_tuple = sim_single_params[idx]
                retry_future = self._executor.submit(_simulate_single, *arg_tuple)
                result = retry_future.result()
                results_all.extend(result)
        self._futures.clear()
        return results_all



def _simulate_single(base_cfg:dict, 
                     server_ip:str, 
                     carla_port:int, 
                     cfg_out_dir:str, 
                     counter, 
                     scenario_name:str, 
                     design:Design, 
                     num_episodes: int, 
                     version: str)->List[EpisodeResult]:
    
    cfg = _build_cfg(scenario_name, base_cfg, version, server_ip, carla_port, design, num_episodes)
    sub_path = os.path.join(cfg_out_dir, str(counter))
    cfg_path = os.path.join(sub_path, f'{scenario_name}{counter}.toml')
    os.makedirs(sub_path, exist_ok=True)
    with open(cfg_path, 'w') as f:
        toml.dump(cfg, f, TomlPreserveInlineDictEncoder())
    try:
        data_out = runner.main(sub_path, cfg_path)
    except (RuntimeError, ConnectionRefusedError, BrokenPipeError, subprocess.CalledProcessError, OSError) as e:
        raise SimInterface.CarlaServerDead(carla_port) from e
    sim_result = []

    for result in data_out:
        try:
            episode_result = _sim_output_to_episode_result(result)
            sim_result.append(episode_result)
        except ValueError as e:
            logger.exception(f"Parsing episode result from port <{carla_port}> of <{cfg_out_dir}> failed with {e}")
            raise
    return sim_result

def _sim_output_to_episode_result(data_out: dict)->EpisodeResult:
    result_dict = output_parser.parse(data_out)

    aoi_perc = result_dict.get("aoi_perc_first_msg_min_dist")
    aoi_avg = result_dict.get("aoi_avg_first_msg_min_dist")
    aoi_peak = result_dict.get("aoi_perc_first_msg_min_dist")
    aoi_95 = aoi_perc.get("p95") if isinstance(aoi_perc, dict) else None
    aoi_99 = aoi_perc.get("p99") if isinstance(aoi_perc, dict) else None
    aoi_100 = aoi_perc.get("p100") if isinstance(aoi_perc, dict) else None
    vel_at_min_d = result_dict.get("ego_vel_at_min_dist")
    min_d_to_target = result_dict.get("min_dist_to_target")
    crash = result_dict.get("col")
    first_msg_t = result_dict.get("first_msg_t")
    execution = result_dict.get("execution", {}) or {}

    result = EpisodeResult()
    result.result["meta"] = {
        "simulator_seed": execution.get("simulation_seed"),
        "root_simulation_seed": execution.get(
            "root_simulation_seed", execution.get("simulation_seed")
        ),
        "simulator_episode_index": execution.get("episode_index"),
        "simulator_trajectory_path": execution.get("simulator_trajectory_path"),
    }
    result.result["kpi"]={
                    "aoi_avg": aoi_avg,
                    "aoi_peak_avg": aoi_peak,
                    "aoi_95": aoi_95,
                    "aoi_99": aoi_99,
                    "aoi_100": aoi_100,
                    "vel_at_min_d": vel_at_min_d,
                    }
    result.result["safety_m"]={
                   "crash": crash,
                   "min_d": min_d_to_target,
                    }
    
    result.result["extra"]={
                    "first_msg_t": first_msg_t,
                    "min_dist_t_ms": result_dict.get("min_dist_t_ms"),
                    "distance_trace": result_dict.get("distance_trace", []),
                    } 
    
    if "rare_event" in result_dict:
        result.result["rare_event"] = result_dict["rare_event"]

    return result

def _build_cfg(scenario_name: str, base_cfg: dict, version: str, server_ip: str, port: int, design: Design,
               num_episodes: int):
    if design["prob"] not in (0.75, 0.975):
        logger.error("Probability detection value is not in (0.75, 0.975)!!")
        raise Exception(f"Probability detection value {design['prob']} is not in (0.75, 0.975)!!")
    if design["prob"] == 0.75:
        probA_det, probA_miss, probB_det, probB_miss = 0.8, 0.2, 0.7, 0.3
    else:
        probA_det, probA_miss, probB_det, probB_miss = 0.99, 0.01, 0.9, 0.1
    cfg = copy.deepcopy(base_cfg)
    cfg['version'] = version
    cfg['name'] = scenario_name
    cfg['server_ip'] = server_ip
    cfg['server_port'] = port
    cfg['sim']['num_episodes'] = num_episodes
    cfg['sim']['delta_time'] = base_cfg['sim']['delta_time']
    # Communication
    if "network.delay" in design and "network.jit" in design:
        cfg['communication']['network']['delay'] = float(design["network.delay"])
        cfg['communication']['network']['jitter'] = float(design["network.jit"])

    if "network.delay_min" in design and "network.delay_avg" in design and "network.jit" in design:
        cfg['communication']['network']['delay_min'] = float(design["network.delay_min"])
        cfg['communication']['network']['delay_avg'] = float(design["network.delay_avg"])
        cfg['communication']['network']['jitter'] = float(design["network.jit"])

    cfg['communication']['fault']['offline'] = math.floor(design["fault"])
    cfg['communication']['network']['packet_drop_rate'] = float(design["packet_drop_rate"])
    # Sensor
    sensor = cfg['sensors'][0]
    sensor['gen_time'] = float(design["gen_t"])
    sensor['fov']['sector']['probA']['det'] = probA_det
    sensor['fov']['sector']['probA']['miss'] = probA_miss
    sensor['fov']['sector']['probA']['wrong'] = 0.0
    sensor['fov']['sector']['probA']['ghost'] = 0.0
    sensor['fov']['sector']['probB']['det'] = probB_det
    sensor['fov']['sector']['probB']['miss'] = probB_miss
    sensor['fov']['sector']['probB']['wrong'] = 0.0
    sensor['fov']['sector']['probB']['ghost'] = 0.0
    # Uncertainty
    sensor['fov']['sector']['uncertainty']['pos'] = float(design["unc_p"])
    sensor['fov']['sector']['uncertainty']['vel'] = float(design["unc_v"])
    # AD stack period
    cfg["ego"]["decision_period"] = float(design["ego_ad_period"])
    return cfg
