# Design-space exploration with audited safe sets

Code to simulate a cooperative system in CARLA, learn safety
surrogates with active learning, audit the candidate set they propose to obtain
the audited set (Phase 1), search the audited set with ParEGO (Phase 2), and plot
the reported results.

Source: [leptarip/coreq](https://github.com/leptarip/coreq).
Data archives and the custom CARLA map: [release v1.0.0](https://github.com/leptarip/coreq/releases/tag/v1.0.0).

## Setup

### Simulator

- Ubuntu 20.04 (or equivalent) with a GPU capable of running CARLA.
- CARLA 0.9.14. Scenario `intersection1` uses `Town05`; `intersection2` uses the
  custom map `sim01`, supplied as a separate release asset. See the
  [map installation instructions](RELEASE_ASSETS.md#custom-carla-map).
- Redis 5.x listening on `localhost:6379`.

### Python

Run the setup and workflow commands from the repository root. Building the
simulator extensions requires a C compiler and the Python 3.8 development
headers.

Two interpreters are used: Python 3.8 for everything that imports CARLA, and
Python 3.11 for the quantile random forest (QRF), ParEGO and the figures.

```bash
python3.8 -m venv .venv38
.venv38/bin/python -m pip install --upgrade pip
.venv38/bin/python -m pip install -r requirements-py38.txt
python3.11 -m venv .venv311
.venv311/bin/python -m pip install --upgrade pip
.venv311/bin/python -m pip install -r requirements-py311.txt
.venv38/bin/python setup_cython.py build_ext --inplace
```

Upgrade pip before installing: the pip bundled with older Python 3.8 installations
may not recognize the CARLA wheel. Cython extensions are needed only by the
Python 3.8 simulator; the Python 3.11 workflows use the simulator through IPC.

CARLA's `agents` package is not on PyPI. It is in the CARLA installation's
`PythonAPI/carla` directory, which must be on the Python path. Set the interpreter
paths and the paths to your CARLA installation:

```bash
export PYTHON38="$PWD/.venv38/bin/python"
export PYTHON311="$PWD/.venv311/bin/python"
export CARLA_PATH=/path/to/CARLA_0.9.14/CarlaUE4.sh
export CARLA_AGENTS_PATH=/path/to/CARLA_0.9.14/PythonAPI/carla
export PYTHONPATH="$PWD:$CARLA_AGENTS_PATH"
```

RF active learning and ParEGO read `CARLA_PATH`; ParEGO also reads
`CARLA_AGENTS_PATH`. The audit freezes its runtime paths from its JSON
configuration, as described below.

### Data

- `results/` is included. It holds the Phase 1 audited set (audit result, scores
  and audited designs), the Phase 2 frontier tables and the analyses behind the figures.
- [Release assets and checksums](RELEASE_ASSETS.md) list the exact downloads,
  disk requirements and installation commands. Verify the archives before use.
- Episode and campaign archives are distributed separately. Place
  `phase1_labelling.tar.gz` and `phase3_campaigns.tar.gz` in the repository root,
  then unpack the archives needed for the workflow you want to run:

```bash
mkdir -p episode_dataset
tar -xzf phase1_labelling.tar.gz -C episode_dataset --strip-components=1
tar -xzf phase3_campaigns.tar.gz
```

The labelling archive has a `phase1_labelling/` wrapper; stripping it places the
shards under `episode_dataset/intersection1/` and `episode_dataset/intersection2/`.
The campaign archive already has the required `phase3_campaigns/` layout.
The separate `phase1_audit` dataset is audit evidence, not training input.

Training and simulation outputs go to `output/` and `output_surrogate_models/`.
The plotting example below writes to `output/`.

## Tests

```bash
$PYTHON38  -m pytest tests
$PYTHON311 -m pytest tests/phase2 tests/phase1/test_qrf_calibration.py
```

## Simulation

One episode against a CARLA server that is already running on the scenario's
`server_port`:

```bash
$PYTHON38 -m source.simulation_environment.runner.runner \
    --config configuration/config_intersection1_debug.toml --out output/simulation
```

The active-learning, audit and ParEGO commands below start and stop their own
CARLA workers.

[`source/simulation_environment/README.md`](source/simulation_environment/README.md)
describes what is being simulated: the two scenarios, the configuration keys,
the sensor, fault and network models, and the episode data the simulator writes.

## Phase 1: safety surrogates and the audited set

Active learning for the RF classifier on one scenario. It warm-starts from the
cached episodes, then alternates between selecting unevaluated designs, running
their episodes and refitting:

```bash
$PYTHON38 -m source.design_exploration.surrogates.rf_cls.launch_rf_al --scenario-name intersection2
```

Train all five families offline on the cached episodes of one scenario
(the builder starts Python 3.11 for QRF using `PYTHON311`):

```bash
$PYTHON38 -m source.design_exploration.tools.build_offline_surrogates_from_cache --scenario-name intersection2
```

The builder creates a fresh labelled dataset and trains all five surrogate families.
Use `--split-path /path/to/offline_splits.json` to reuse existing splits and labels;
the scenario, design space and safety/SPRT settings must match the saved dataset.
Reused split and label files are preserved. Use `--max-train-workers 1` for sequential
training, or `--skip-qrf` to train only the four other families.


Repeat with `--scenario-name intersection1` to train the other scenario.
This command fits models from the supplied cache; it does not launch CARLA.

Portfolio audit of a score vector. `freeze` fixes the thresholds, the sampling
plan and the seeds without starting CARLA; `execute` runs the frozen trials.
The configuration in `reproduction/phase1/audit_config/` audits the shipped
scores; it does not automatically use the models just trained. Set
`runtime.carla_path` and `runtime.carla_agents_path` in `audit.json` before
freezing. `preflight` checks the frozen sets and sampling plans offline; it does
not test CARLA, Redis or map availability.

```bash
$PYTHON38 -m reproduction.phase1.run_audit freeze    --config reproduction/phase1/audit_config/audit.json
$PYTHON38 -m reproduction.phase1.run_audit preflight --freeze-root output/phase1/audit_freeze
$PYTHON38 -m reproduction.phase1.run_audit execute   --freeze-root output/phase1/audit_freeze
# Set this to the directory printed by execute after "Audit complete:".
AUDIT_RUN="output/phase1/audit_run/audit_YYYYMMDD_HHMMSS_ID"
$PYTHON38 -m reproduction.phase1.finalize_audit \
    --freeze-root output/phase1/audit_freeze --run-dir "$AUDIT_RUN" --out-root output/phase1/audited_set
```

`freeze` refuses to overwrite an existing freeze directory. For a new audit,
choose a new `--freeze-root` and use it consistently in the subsequent commands.
To continue an interrupted execution, pass its directory with
`execute --freeze-root output/phase1/audit_freeze --resume "$AUDIT_RUN"`.

## Phase 2: ParEGO over the audited set

The launcher reads the audited set in `results/phase1/audited_set/` (or `--audited-set DIR`),
freezes the run in `output/phase2/<run>/run_manifest.json` and runs 10
random plus 20 acquired designs with 20 episodes per scenario. To use a newly
finalized audit, pass `--audited-set output/phase1/audited_set`.
`--prepare-only` writes a run manifest without starting CARLA; the following
command without that flag creates and executes a separate run.

```bash
$PYTHON38 -m source.design_exploration.optimization.launch_parego --prepare-only   # validate only
$PYTHON38 -m source.design_exploration.optimization.launch_parego
```

Other starts use a different optimizer seed and a disjoint episode-seed range:

```bash
$PYTHON38 -m source.design_exploration.optimization.launch_parego --random-seed 23 --episode-seed-offset 500000000
$PYTHON38 -m source.design_exploration.optimization.launch_parego --random-seed 37 --episode-seed-offset 600000000
```

`--resume-run-dir output/phase2/<run>` continues an interrupted run, and
`--cost-profile-id` selects another profile from `configuration/cost_profiles.json`.

Generate the frontier figure:

```bash
$PYTHON311 -m reproduction.phase2.plot_frontier --out-dir output/phase2/figures
```

## Phase 3: rare-event estimators

The estimators and importance-sampling proposals are in
`source/design_exploration/rare_events/`; the analyses of the reported
estimates are in `results/phase3/`.

A campaign writes one record per episode into `sample-shards`, carrying that
episode's likelihood ratio, its failure indicator and the route features that
classify it. The reported probabilities are a weighted aggregation of those
records, so they can be recomputed without a simulator. Unpack the campaign
archive and replay either campaign:

```bash
$PYTHON38 -m reproduction.phase3.replay_campaign \
  --campaign phase3_campaigns/frontier_v3_6499_family_10000_v1 \
  --published results/phase3/design_6499_campaign_analysis.json
$PYTHON38 -m reproduction.phase3.replay_campaign \
  --campaign phase3_campaigns/frontier_v3_efficiency_15h_v2 \
  --published results/phase3/efficiency_campaign_analysis.json
```

It verifies each shard against its digest, recomputes the principal mode and the
three route families under the scheme each job was run with, and exits non-zero
if any endpoint differs from the published file.

The shard contents themselves are taken as given. Route signatures and
likelihood ratios are derived from the simulator logs, and re-deriving them is
the campaign's own audit, which needs the trajectories rather than the shards.
Phase 1 includes the audited set and its statistics in the repository. Its
historical episode outcomes are available separately in `phase1_audit.tar.gz`;
re-running the audit against CARLA creates fresh evidence.

## Layout

| Path | Contents |
| --- | --- |
| `source/simulation_environment/` | CARLA scenarios, agents, off-board perception, communication, logging |
| `source/design_exploration/` | Design space, episode store, SPRT labelling, surrogates, audit, ParEGO, rare-event estimators |
| `reproduction/` | Scripts that regenerate the reported tables and figures |
| `configuration/` | Scenario configurations and cost profiles |
| `results/` | Audited set, frontier tables, analyses and figures |
| `tests/` | Test suite |

## License

Project code is licensed under [Apache License 2.0](LICENSE), except for the
MIT-licensed portions identified in source headers:

- The CARLA-derived Cython files `basic_agent_cy.pyx`, `controller_cy.pyx`,
  `global_route_planner_cy.pyx`, `local_planner_cy.pyx`, and `misc_cy.pyx` in
  `source/simulation_environment/cython/` retain their upstream MIT notices.

The [MIT license text](LICENSES/MIT.txt) accompanies those notices. Upstream
copyright and author attributions are retained in the corresponding files.
