#!/usr/bin/env python3
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
NDJSON simulation server intended to run under Python 3.8.

Protocol:

Request:
    {
        "id": "<request id>",
        "action": "evaluate_design",
        "design": { ... },
        "scenario_seeds": [
            {"name": "<scenario_name>", "seeds": [101, 102, 103]},
            ...
        ]
    }

Response:
    {
        "id": "<request id>",
        "ok": true,
        "episodes": {
            "<scenario_name>": [
                {"seed": 101, "payload": {...}},
                ...
            ]
        }
    }
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from source.design_exploration.commons.design_space import Design

# These Python-3.8/CARLA dependencies are loaded only when a live server is
# constructed. Pure scheduling/checkpoint tests can import this module without
# importing the simulator extension.
cfg_parser = None
CarlaWorkerPool = None

LOG = logging.getLogger("ipc.sim_server")


def _parse_scenario_arg(arg: str):
    if ":" not in arg:
        raise argparse.ArgumentTypeError("Expected format <name>:<cfg_path>")
    name, cfg_path = arg.split(":", 1)
    return name, cfg_path


class PoolBackedSimServer:
    def __init__(
        self,
        scenario_cfgs: Dict[str, str],
        algo_version: str,
        sim_out: str,
        num_workers: int,
        carla_path: str,
        episode_chunk_size: int = 1,
        max_in_flight: Optional[int] = None,
    ):
        global cfg_parser, CarlaWorkerPool
        if cfg_parser is None:
            import source.simulation_environment.cfg.cfg_parser as cfg_parser_module

            cfg_parser = cfg_parser_module
        if CarlaWorkerPool is None:
            from source.design_exploration.simulator_interface.carla_worker_pool import (
                CarlaWorkerPool as pool_class,
            )

            CarlaWorkerPool = pool_class
        self.scenario_cfgs = dict(scenario_cfgs)
        self.algo_version = algo_version
        self.sim_out = sim_out
        self.num_workers = max(1, int(num_workers))
        self.episode_chunk_size = max(1, int(episode_chunk_size))
        self.max_in_flight = max(
            self.num_workers,
            int(max_in_flight) if max_in_flight is not None else 2 * self.num_workers,
        )
        self.checkpoint_root = Path(self.sim_out) / "ipc_request_checkpoints"
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)

        helpers = {}
        for name, cfg_path in self.scenario_cfgs.items():
            helpers[name] = cfg_parser.parse(cfg_path)
        self.helpers = helpers
        self.pool = CarlaWorkerPool(
            num_workers=self.num_workers,
            algo_version=self.algo_version,
            output_root=self.sim_out,
            carla_path=carla_path,
        )
        time.sleep(0.5)
        if self.pool.alive_workers <= 0:
            raise RuntimeError("No CARLA workers are alive after IPC server startup.")

    def close(self):
        self.pool.shutdown()

    def _error_response(self, req_id: str, msg: str):
        return {"id": req_id, "ok": False, "error": msg}

    @staticmethod
    def _split_fixed(values: List[int], chunk_size: int) -> List[List[int]]:
        if not values:
            return []
        chunk_size = max(1, int(chunk_size))
        return [values[start:start + chunk_size] for start in range(0, len(values), chunk_size)]

    def _effective_chunk_size(self, scenario_plans: List[Dict[str, Any]]) -> int:
        """Shrink the configured chunk size until every worker gets a job.

        `_split_fixed` alone can hand three workers two chunks -- two scenarios at
        `--episode-chunk-size 20` with 20 seeds each -- leaving a worker idle for
        the whole design evaluation. Never grows the chunk beyond what was asked
        for, and is a no-op at the default size of 1.
        """
        total_seeds = sum(len(plan["seeds"]) for plan in scenario_plans)
        if total_seeds <= 0:
            return self.episode_chunk_size
        configured = max(1, int(self.episode_chunk_size))
        chunks_at_configured = -(-total_seeds // configured)  # ceil
        desired_chunks = max(self.num_workers, chunks_at_configured)
        desired_chunks = min(desired_chunks, total_seeds)
        return max(1, -(-total_seeds // desired_chunks))

    def _plan_chunks(self, scenario_plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        chunk_size = self._effective_chunk_size(scenario_plans)
        for plan in scenario_plans:
            seed_chunks = self._split_fixed(plan["seeds"], chunk_size)
            for chunk_idx, chunk_seeds in enumerate(seed_chunks):
                chunks.append(
                    {
                        "name": plan["name"],
                        "cfg_base": copy.deepcopy(plan["cfg_base"]),
                        "seeds": chunk_seeds,
                        "chunk_index": chunk_idx,
                    }
                )
        return chunks

    @staticmethod
    def _request_key(design: Dict[str, Any], scenario_plans: List[Dict[str, Any]]) -> str:
        identity = {
            "design": design,
            "scenario_seeds": [
                {"name": plan["name"], "seeds": [int(seed) for seed in plan["seeds"]]}
                for plan in scenario_plans
            ],
            "protocol": "evaluate_design_v3",
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _chunk_key(self, design: Dict[str, Any], chunk: Dict[str, Any]) -> str:
        identity = {
            "design": design,
            "scenario": chunk["name"],
            "seeds": [int(seed) for seed in chunk["seeds"]],
            "algo_version": self.algo_version,
            "scenario_cfg_sha256": self._scenario_cfg_digest(chunk["name"]),
            "protocol": "evaluate_design_chunk_v2",
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _scenario_cfg_digest(self, scenario: str) -> str:
        """Digest the parsed scenario config so edited configs cannot be resumed into."""
        encoded = json.dumps(
            self.helpers.get(scenario, {}),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _episode_key(self, design: Dict[str, Any], scenario: str, seed: int) -> str:
        """Identify one episode independently of how it happened to be chunked.

        Keying on the chunk's whole seed list made every checkpoint unreadable
        after `--episode-chunk-size` changed, silently re-simulating the run.
        `algo_version` and the scenario config are folded in so a checkpoint
        written under different settings simply does not match, rather than
        being restored and mixed into a fresh campaign.
        """
        identity = {
            "design": design,
            "scenario": scenario,
            "seed": int(seed),
            "algo_version": self.algo_version,
            "scenario_cfg_sha256": self._scenario_cfg_digest(scenario),
            "protocol": "evaluate_design_episode_v1",
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _episode_checkpoint_path(self, episode_key: str) -> Path:
        return self.checkpoint_root / episode_key[:2] / f"{episode_key}.json"

    def _chunk_checkpoint_path(self, chunk: Dict[str, Any]) -> Path:
        chunk_key = str(chunk["checkpoint_key"])
        return self.checkpoint_root / chunk_key[:2] / f"{chunk_key}.json"

    def _chunk_failure_path(self, chunk: Dict[str, Any]) -> Path:
        path = self._chunk_checkpoint_path(chunk)
        return path.with_name(f"{path.stem}.failures.json")

    @staticmethod
    def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp_path), str(path))

    def _load_chunk_checkpoint(
        self, request_key: str, chunk: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[int], List[str], Dict[str, int]]:
        """Restore valid episodes independently of request shape and chunking.

        ``request_key`` is retained as provenance, but it cannot be an identity
        constraint: the optimizer deliberately retries only missing scenarios
        after a crash between scenario-store appends. Episode identity already
        binds design, scenario, seed, algorithm, and parsed scenario config.
        """
        items: List[Dict[str, Any]] = []
        missing_seeds: List[int] = []
        missing_keys: List[str] = []
        attempts_by_origin: Dict[str, int] = {}
        for seed, episode_key in zip(chunk["seeds"], chunk["episode_keys"]):
            path = self._episode_checkpoint_path(episode_key)
            if not path.is_file():
                missing_seeds.append(int(seed))
                missing_keys.append(episode_key)
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    record = json.load(handle)
                if int(record.get("schema_version", -1)) != 3:
                    raise ValueError("unsupported checkpoint schema")
                if record.get("episode_key") != episode_key:
                    raise ValueError("episode key mismatch")
                if record.get("scenario") != chunk["name"]:
                    raise ValueError("scenario mismatch")
                if int(record.get("seed", -1)) != int(seed):
                    raise ValueError("seed mismatch")
                if record.get("algo_version") != self.algo_version:
                    raise ValueError("algorithm mismatch")
                if record.get("scenario_cfg_sha256") != self._scenario_cfg_digest(chunk["name"]):
                    raise ValueError("scenario config mismatch")
                payload = record.get("payload")
                if not isinstance(payload, dict):
                    raise ValueError("payload is not an object")
                items.append({"seed": int(seed), "payload": payload})
                origin = str(record.get("origin_job_key") or record.get("chunk_key") or episode_key)
                attempts_by_origin[origin] = max(
                    attempts_by_origin.get(origin, 0), int(record.get("attempts", 1))
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                LOG.warning("Ignoring invalid IPC episode checkpoint: %s", path)
                missing_seeds.append(int(seed))
                missing_keys.append(episode_key)
        return items, missing_seeds, missing_keys, attempts_by_origin

    def _save_chunk_checkpoint(
        self,
        request_key: str,
        chunk: Dict[str, Any],
        items: List[Dict[str, Any]],
        attempts: int,
    ) -> None:
        for item, episode_key in zip(items, chunk["episode_keys"]):
            payload = item["payload"]
            record = {
                "schema_version": 3,
                "request_key": request_key,
                "episode_key": episode_key,
                "chunk_key": chunk["checkpoint_key"],
                "origin_job_key": str(
                    chunk.get("origin_job_key") or chunk["checkpoint_key"]
                ),
                "scenario": chunk["name"],
                "algo_version": self.algo_version,
                "scenario_cfg_sha256": self._scenario_cfg_digest(chunk["name"]),
                "seed": int(item["seed"]),
                "attempts": int(attempts),
                "payload": payload,
            }
            self._write_json_atomic(
                self._episode_checkpoint_path(episode_key),
                record,
            )

    def _load_chunk_failures(self, chunk: Dict[str, Any]) -> Tuple[int, int]:
        path = self._chunk_failure_path(chunk)
        if not path.is_file():
            return 0, 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            if record.get("chunk_key") != chunk["checkpoint_key"]:
                return 0, 0
            return (
                int(record.get("failed_worker_attempts", 0)),
                int(record.get("terminal_failures", 0)),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            LOG.warning("Ignoring invalid IPC chunk failure journal: %s", path)
            return 0, 0

    def _record_chunk_failure(self, chunk: Dict[str, Any], attempts: int) -> None:
        prior_attempts, prior_failures = self._load_chunk_failures(chunk)
        self._write_json_atomic(
            self._chunk_failure_path(chunk),
            {
                "schema_version": 2,
                "chunk_key": chunk["checkpoint_key"],
                "scenario": chunk["name"],
                "seeds": [int(seed) for seed in chunk["seeds"]],
                "episode_keys": list(chunk["episode_keys"]),
                "algo_version": self.algo_version,
                "scenario_cfg_sha256": self._scenario_cfg_digest(chunk["name"]),
                "failed_worker_attempts": prior_attempts + int(attempts),
                "terminal_failures": prior_failures + 1,
            },
        )

    def _load_request_failures(self, chunks: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Reconcile failure journals even when a resumed request is re-chunked."""
        requested_keys = {
            key for chunk in chunks for key in chunk.get("episode_keys", [])
        }
        paths = {self._chunk_failure_path(chunk) for chunk in chunks}
        paths.update(self.checkpoint_root.rglob("*.failures.json"))
        attempts = 0
        failures = 0
        for path in paths:
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    record = json.load(handle)
                if int(record.get("schema_version", -1)) != 2:
                    continue
                episode_keys = set(record.get("episode_keys") or [])
                scenario = str(record.get("scenario", ""))
                if not episode_keys or not episode_keys <= requested_keys:
                    continue
                if record.get("algo_version") != self.algo_version:
                    continue
                if record.get("scenario_cfg_sha256") != self._scenario_cfg_digest(scenario):
                    continue
                attempts += int(record.get("failed_worker_attempts", 0))
                failures += int(record.get("terminal_failures", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                LOG.warning("Ignoring invalid IPC chunk failure journal: %s", path)
        return attempts, failures

    @staticmethod
    def _payload_to_dict(payload: Any) -> Dict[str, Any]:
        if hasattr(payload, "result"):
            result_dict = getattr(payload, "result", {}) or {}
            return dict(result_dict) if isinstance(result_dict, dict) else {"payload": result_dict}
        if isinstance(payload, dict):
            return dict(payload)
        return {"payload": payload}

    @staticmethod
    def _inject_seed(payload: Dict[str, Any], seed: int) -> Dict[str, Any]:
        out = dict(payload)
        meta = dict(out.get("meta", {}) or {})
        meta["seed"] = int(seed)
        out["meta"] = meta
        return out

    def _collect_results(
        self,
        req_id: str,
        request_key: str,
        chunks: List[Dict[str, Any]],
        scenario_plans: List[Dict[str, Any]],
        design_dict: Dict[str, Any],
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
        results_by_scenario: Dict[str, Dict[int, List[Dict[str, Any]]]] = {
            plan["name"]: {} for plan in scenario_plans
        }

        queued: List[Dict[str, Any]] = []
        resumed_chunks = 0
        partially_resumed_chunks = 0
        resumed_episodes = 0
        resumed_attempts_by_origin: Dict[str, int] = {}
        prior_failed_worker_attempts, prior_terminal_failures = (
            self._load_request_failures(chunks)
        )
        for chunk in chunks:
            items, missing_seeds, missing_keys, origin_attempts = self._load_chunk_checkpoint(
                request_key, chunk
            )
            if items:
                results_by_scenario[chunk["name"]][chunk["chunk_index"]] = items
                resumed_episodes += len(items)
                for origin, attempts in origin_attempts.items():
                    resumed_attempts_by_origin[origin] = max(
                        resumed_attempts_by_origin.get(origin, 0), attempts
                    )
            if not missing_seeds:
                resumed_chunks += 1
            else:
                if items:
                    partially_resumed_chunks += 1
                missing_chunk = dict(chunk)
                missing_chunk["seeds"] = missing_seeds
                missing_chunk["episode_keys"] = missing_keys
                queued.append(missing_chunk)

        resumed_recorded_attempts = sum(resumed_attempts_by_origin.values())

        pending: Dict[str, Dict[str, Any]] = {}
        abort_error: Optional[str] = None
        submitted_chunks = 0
        submitted_episodes = 0
        submitted_worker_attempts = 0
        retried_chunks = 0

        def submit_available() -> None:
            nonlocal submitted_chunks, submitted_episodes
            while queued and len(pending) < self.max_in_flight:
                chunk = queued.pop(0)
                job_id = self.pool.submit(
                    scenario_name=chunk["name"],
                    scenario_cfg=chunk["cfg_base"],
                    design=design_dict,
                    seeds=chunk["seeds"],
                    num_episodes=len(chunk["seeds"]),
                )
                chunk["origin_job_key"] = f"{request_key}:{job_id}"
                pending[job_id] = chunk
                submitted_chunks += 1
                submitted_episodes += len(chunk["seeds"])

        submit_available()

        while pending:
            if self.pool.fatal_error and abort_error is None:
                abort_error = str(self.pool.fatal_error)
                queued.clear()
            completed = self.pool.poll()
            if self.pool.fatal_error and abort_error is None:
                abort_error = str(self.pool.fatal_error)
                queued.clear()
            if not completed:
                if self.pool.alive_workers <= 0:
                    if abort_error is None:
                        abort_error = "All CARLA workers are dead."
                    break
                time.sleep(0.05)
                continue

            for rec in completed:
                job = pending.pop(rec.job_id, None)
                if job is None:
                    continue
                if rec.status != "done":
                    self._record_chunk_failure(
                        job, int(getattr(rec, "attempts", 1))
                    )
                    # Do not raise here. Sibling jobs in this same batch have
                    # already produced episodes, and more are still running; both
                    # were previously discarded, so a single CARLA death threw
                    # away up to `max_in_flight` completed episodes. Stop feeding
                    # the pool, let what is in flight land and be checkpointed,
                    # then fail the request.
                    if abort_error is None:
                        abort_error = (
                            f"Worker job failed for request {req_id} "
                            f"scenario '{job['name']}': {rec.error}"
                        )
                    queued.clear()
                    continue

                attempts_for_chunk = int(getattr(rec, "attempts", 1))
                submitted_worker_attempts += attempts_for_chunk
                if attempts_for_chunk > 1:
                    retried_chunks += 1

                payload_list = rec.payload if isinstance(rec.payload, list) else [rec.payload]
                if len(payload_list) != len(job["seeds"]):
                    self._record_chunk_failure(job, attempts_for_chunk)
                    if abort_error is None:
                        abort_error = (
                            f"Worker returned {len(payload_list)} payloads for scenario "
                            f"'{job['name']}', expected {len(job['seeds'])}."
                        )
                    queued.clear()
                    continue

                items = []
                for seed_val, payload in zip(job["seeds"], payload_list):
                    payload_dict = self._inject_seed(self._payload_to_dict(payload), int(seed_val))
                    items.append({"seed": int(seed_val), "payload": payload_dict})
                results_by_scenario[job["name"]].setdefault(job["chunk_index"], []).extend(items)
                self._save_chunk_checkpoint(
                    request_key, job, items, attempts_for_chunk
                )
            if abort_error is None:
                submit_available()

        if abort_error is not None:
            raise RuntimeError(abort_error)

        episodes_map: Dict[str, List[Dict[str, Any]]] = {}
        for plan in scenario_plans:
            requested = [int(seed) for seed in plan["seeds"]]
            by_seed: Dict[int, Dict[str, Any]] = {}
            for _, chunk_items in sorted(results_by_scenario[plan["name"]].items()):
                for item in chunk_items:
                    seed_val = int(item["seed"])
                    if seed_val in by_seed:
                        raise RuntimeError(
                            f"Duplicate seed {seed_val} returned for scenario '{plan['name']}'."
                        )
                    by_seed[seed_val] = item
            missing = [seed for seed in requested if seed not in by_seed]
            if missing:
                raise RuntimeError(
                    f"Scenario '{plan['name']}' is missing payloads for seeds {missing}."
                )
            episodes_map[plan["name"]] = [by_seed[seed] for seed in requested]
        return episodes_map, {
            "planned_chunks": int(len(chunks)),
            "resumed_chunks": int(resumed_chunks),
            "partially_resumed_chunks": int(partially_resumed_chunks),
            "resumed_episodes": int(resumed_episodes),
            "submitted_chunks": int(submitted_chunks),
            "submitted_episodes": int(submitted_episodes),
            "submitted_worker_attempts": int(submitted_worker_attempts),
            "retried_chunks": int(retried_chunks),
            "resumed_recorded_attempts": int(resumed_recorded_attempts),
            "prior_failed_worker_attempts": int(prior_failed_worker_attempts),
            "prior_terminal_failures": int(prior_terminal_failures),
            "total_recorded_worker_attempts": int(
                resumed_recorded_attempts
                + prior_failed_worker_attempts
                + submitted_worker_attempts
            ),
            "episode_chunk_size": int(self.episode_chunk_size),
            # What the planner actually used. `_effective_chunk_size` shrinks the
            # configured size when it would leave a worker idle, so reporting the
            # configured value alone would misdescribe the run.
            "effective_chunk_size": int(
                max((len(chunk["seeds"]) for chunk in chunks), default=0)
            ),
            "max_in_flight": int(self.max_in_flight),
        }

    def _handle_evaluate_design(self, req: dict):
        req_id = req.get("id", "")
        design_dict = req.get("design")
        if not isinstance(design_dict, dict):
            return self._error_response(req_id, "design must be a dict")

        raw_scenarios = req.get("scenario_seeds")
        if not isinstance(raw_scenarios, list) or not raw_scenarios:
            return self._error_response(req_id, "scenario_seeds must be a non-empty list")

        try:
            Design(design_dict)
        except Exception as exc:
            return self._error_response(req_id, f"invalid design: {exc}")

        scenario_plans: List[Dict[str, Any]] = []
        seen_names = set()
        for entry in raw_scenarios:
            if not isinstance(entry, dict):
                return self._error_response(req_id, "scenario_seeds entries must be dicts")
            name = entry.get("name")
            if not isinstance(name, str) or name not in self.helpers:
                return self._error_response(req_id, f"unknown scenario '{name}'")
            if name in seen_names:
                return self._error_response(req_id, f"duplicate scenario '{name}' in request")
            seen_names.add(name)
            seed_list_raw = entry.get("seeds")
            if not isinstance(seed_list_raw, list) or not seed_list_raw:
                return self._error_response(req_id, f"scenario '{name}' must provide a non-empty seeds list")
            try:
                seed_list = [int(seed) for seed in seed_list_raw]
            except Exception:
                return self._error_response(req_id, f"scenario '{name}' has a non-integer seed")
            if len(set(seed_list)) != len(seed_list):
                return self._error_response(req_id, f"scenario '{name}' contains duplicate seeds")
            scenario_plans.append(
                {
                    "name": name,
                    "cfg_base": copy.deepcopy(self.helpers[name]),
                    "seeds": seed_list,
                }
            )

        try:
            chunks = self._plan_chunks(scenario_plans)
            request_key = self._request_key(design_dict, scenario_plans)
            for chunk in chunks:
                chunk["checkpoint_key"] = self._chunk_key(design_dict, chunk)
                chunk["episode_keys"] = [
                    self._episode_key(design_dict, chunk["name"], seed)
                    for seed in chunk["seeds"]
                ]
            episodes_map, accounting = self._collect_results(
                req_id, request_key, chunks, scenario_plans, design_dict
            )
        except Exception as exc:
            LOG.exception("evaluate_design failed")
            return self._error_response(req_id, f"evaluate_design error: {exc}")

        return {
            "id": req_id,
            "ok": True,
            "request_key": request_key,
            "episodes": episodes_map,
            "accounting": accounting,
        }

    def serve(self):
        print(
            json.dumps(
                {
                    "ready": True,
                    "protocol": "evaluate_design_v3",
                    "workers": int(self.pool.alive_workers),
                    "episode_chunk_size": int(self.episode_chunk_size),
                    "max_in_flight": int(self.max_in_flight),
                }
            ),
            flush=True,
        )
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except Exception:
                    LOG.error("failed to parse request line: %s", line)
                    continue

                req_id = req.get("id", "")
                action = req.get("action")
                if action == "shutdown":
                    print(json.dumps({"id": req_id, "ok": True, "status": "bye"}), flush=True)
                    break
                if action != "evaluate_design":
                    print(
                        json.dumps(self._error_response(req_id, f"unknown action '{action}'")),
                        flush=True,
                    )
                    continue

                resp = self._handle_evaluate_design(req)
                print(json.dumps(resp), flush=True)
        finally:
            try:
                self.close()
            except Exception:
                traceback.print_exc()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="NDJSON simulation server (Python 3.8)")
    parser.add_argument("--scenario", action="append", required=True, help="<name>:<cfg_path>", dest="scenarios")
    parser.add_argument("--sim-out", default=os.path.join("optimization_results", "sim_out_ipc"))
    parser.add_argument("--algo-version", default="parego_qrf_ipc")
    parser.add_argument("--num-carla", type=int, default=1, help="number of CARLA worker processes")
    parser.add_argument(
        "--episode-chunk-size",
        type=int,
        default=1,
        help="maximum episodes in one retryable worker job (default: 1)",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="maximum queued/running worker jobs (default: 2 * workers)",
    )
    parser.add_argument(
        "--carla-path",
        required=True,
        help="CARLA executable launched by each worker",
    )
    args = parser.parse_args()

    scenario_cfgs = dict(_parse_scenario_arg(s) for s in args.scenarios)
    server = PoolBackedSimServer(
        scenario_cfgs,
        args.algo_version,
        args.sim_out,
        args.num_carla,
        args.carla_path,
        args.episode_chunk_size,
        args.max_in_flight,
    )

    def _terminate_cleanly(signum, _frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGINT, _terminate_cleanly)
    signal.signal(signal.SIGTERM, _terminate_cleanly)
    server.serve()


if __name__ == "__main__":
    main()
