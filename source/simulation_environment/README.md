# The simulation

Each episode places an ego vehicle (role `hero`) and an adversary at a CARLA
intersection. A roadside sensor, the off-board perception system (OBPS),
observes the adversary and sends measurement messages to the ego over a
stochastic communication channel. The ego runs a cooperative-awareness stack
that combines those messages with its own on-board perception and decides
whether to brake or proceed.

An episode ends when the ego completes its route (`agent_done`) or when the
route polygons of the two vehicles intersect (`col_shapes`, a predicted
collision). It writes one JSON file per episode.

## Scenarios

A scenario fixes the map, the actors, their routes, and the termination
conditions. Two are included:

| Scenario | Conflict | Map |
| --- | --- | --- |
| `intersection1` | left turn across path: ego goes straight, adversary turns left across it | `Town05` |
| `intersection2` | straight crossing path, with different sensor placement and timing | `sim01` |

### Adding a scenario

1. Create a package under `scenario/`, following `scenario1/` or `scenario2/`.
   At minimum it needs a `scenario_<N>_cfg_actors.py` with spawn transforms,
   destinations and sensor position helpers, and a `scenario_<N>.py` with a
   class extending `BaseScenario` that implements `generate_scenario()`,
   `pre_run()`, `run_step()`, `is_scenario_done()` and `get_scenario_map()`.
2. Register it in `scenario/scenarios.py`, in both `get_scenario()` and
   `get_scenario_map()`.
3. Add a TOML config under `configuration/` with `scenario = "<name>"`.

## On-board perception is 2D

The ego's field of view is computed by casting rays in the ground plane against
the 2D footprints of vehicles and buildings (`Perception2Dgt`). Dropping the
vertical axis has two consequences worth knowing:

- **Building height is discarded.** A footprint comes from the x/y of the
  bounding box. CARLA reports a multi-storey building as one `EnvironmentObject`
  per floor, all sharing that box, so the floors collapse onto one footprint.
  The duplicates are dropped when footprints are built, since they carry no
  information but would enlarge the spatial index and create ties for the
  nearest obstacle.
- **Only yaw is applied.** CARLA bounding boxes are oriented.
  `create_bb_points_2D` rotates the corners about the box origin by yaw alone:
  pitch and roll would tilt the footprint out of the ground plane this model
  works in. Vehicle footprints come from `create_bb_shape`, which applies the
  full actor transform.

## Configuration

A run is driven by a TOML file; `configuration/` holds one per scenario.

**Top level:** `name` (a label), `scenario`, `server_ip` and `server_port` for
the CARLA server, and `version` for the config schema.

**`[sim]`**

| Key | Meaning |
| --- | --- |
| `delta_time` | simulation step in seconds, e.g. `0.01` for 10 ms |
| `num_episodes` | episodes to run |
| `seed` | optional base seed; taken from the clock when absent |

**`[ego]` and `[adversary]`** share a schema:

| Key | Meaning |
| --- | --- |
| `role_name` | CARLA actor role, `hero` for the ego |
| `decision_period` | how often the agent stack runs, in ms |
| `initial_v`, `target_v` | initial and cruise speed, km/h |
| `perception.fov.degrees` | field-of-view aperture, 30–360 |
| `perception.fov.ray_step` | angular step between rays, degrees |
| `perception.fov.range` | perception range, metres |
| `perception.prediction_window` | prediction horizon, seconds |

**`[communication]`** configures the channel between sensor and ego, in two
sub-tables. `communication.fault.*` takes `type`, `offline` (how long the
channel stays down, in ms, `0` to disable), `trigger_mean` and `trigger_var`
for the Gaussian that samples when it goes down, and an optional `trigger` to
fix that time. `communication.network.*` takes `type`, `packet_drop_rate`, and
the delay parameters its model needs.

**`[[sensors]]`** is an array, one entry per roadside sensor, each with a unique
`name`:

| Key | Meaning |
| --- | --- |
| `recipient` | which vehicle receives the messages |
| `stream` | Redis stream topic |
| `position`, `orientation` | mounting position on the map, and pointing direction in degrees |
| `gen_time` | message generation period, ms |
| `retransmission.queue_size`, `.policy` | retransmission queue depth and eviction policy |
| `fov.type` | sensor field-of-view model |
| `fov.sector.degrees`, `.radius` | aperture and range of the detection sector |
| `fov.sector.scale` | size of the inner region relative to the outer one |
| `fov.sector.probA.*`, `.probB.*` | detection outcome probabilities per region |
| `fov.sector.uncertainty.type`, `.pos`, `.vel` | measurement error model and its magnitudes |

**`[rare_event]`** is optional. With `enabled = true` the simulator draws the
stochastic processes from biased proposals and records log importance weights
in the output. With `enabled = false` every bias sub-table is inert. A sensor,
fault or network model rejects a non-empty bias configuration unless an
importance sampler is supplied, so biased draws cannot be recorded as though
they were nominal.

## The models

### Faults

`sit`, service interrupt time, takes the link offline once per episode. The
trigger time is drawn from a Gaussian set by `trigger_mean` and `trigger_var`,
or fixed with `trigger`; the link returns after `offline` ms. Setting
`offline = 0` keeps it up for the whole episode. Under importance sampling the
trigger is drawn from a biased proposal, with the nominal Gaussian retained for
the weight.

### Network

`log_norm` draws each packet's end-to-end delay as `delay_min` plus a
log-normal variate, whose parameters are fitted so the variable part has mean
`delay_avg - delay_min` and standard deviation `jitter`. `uniform` draws it
uniformly from `[delay, delay + jitter]`. Both drop packets as independent
Bernoulli trials at `packet_drop_rate`.

### Sensor

`sector` models a sector-shaped field of view split into two concentric
regions: an inner region A, and an outer annulus B with its own probabilities.
At each tick the target's 2D box is tested against both. Inside a region the
outcome is sampled as a detection, a miss, or a detection at a wrong position.
Outside both, a ghost detection may fire independently per region.

Measurement error has two forms, set by `uncertainty.type`:

- `worst_case` — a deterministic, **signed** calibration bias added to every
  reading. Negative values bias in the opposite direction; they are not clamped.
- `normal` — zero-mean Gaussian noise whose standard deviation is the
  configured magnitude, which must therefore be non-negative.

`sector_ghost` is a variant whose ghost always appears at a fixed map location
rather than near the target, modelling a persistent infrastructure artefact.

## Output

Each episode writes into the run folder:

```text
<output_root>/<experiment_name>/
  sim_data_<episode>.json    the episode
  log_<episode>.txt          its text log
episodes_progress.txt        resumption checkpoint
```

`sim_data_<N>.json` is one object at `"version": 6`, with `scenario`, `ego`,
`adv`, `obps` and `comm`, plus `execution` and `rare_event` when those apply.

`scenario` carries the name, the start and end time in simulation ms, the step,
and `cause`, the termination reason. `ego` and `adv` share a structure: `data`
holds per-tick kinematics (speed, acceleration and its components, velocity and
position, forward and right vectors, distance to the nearest obstacle, the
planner's selected speed, and the speed the vehicle must not exceed to avoid
the most critical obstacle); `perception_data` holds per-decision distances to
each tracked target; `tactical_data` holds the decisions themselves, each with
the age of information of the message that triggered it, the action taken, the
message identity, and the times to reach and leave the critical region for both
vehicles. `obps` and `comm` hold the sensor's measurements and the channel's
per-packet fate.

### Parsed features

`parsing_output/output_parser.py` summarizes each episode with these features:

| Feature | Meaning |
| --- | --- |
| `col` | whether the episode ended in collision |
| `first_msg_t` | when the ego received its first message, ms |
| `min_dist_to_target` | closest the two vehicles came, m |
| `ego_vel_at_min_dist`, `adv_vel_at_min_dist` | their speeds at that moment, m/s |
| `aoi_avg_first_last_msg`, `aoi_perc_first_last_msg`, `aoi_peak_avg_first_last_msg` | age of information over the message window: mean, percentiles, and the mean of its local peaks |
| `aoi_*_first_msg_min_dist` | the same three, from the first message to the closest approach |
