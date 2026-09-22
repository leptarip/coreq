# Release assets

The data and custom map are separate from the source tree. Download the archives
needed for your workflow from [release v1.0.0](https://github.com/leptarip/coreq/releases/tag/v1.0.0)
under **Assets**. Use the same release version for source and data.
The filenames below are exact; `dirty` is part of the supplied map package's name.
The source can be obtained from the release's `simreq-source.tar.gz` snapshot.

| Asset | Download | Unpacked files | Use |
| --- | ---: | ---: | --- |
| `phase1_audit.tar.gz` | 2.9 MiB | 3.0 MiB | Historical audit evidence; keep separate from training |
| `phase1_labelling.tar.gz` | 12.4 MiB | 12.6 MiB | SPRT labelling and offline training |
| `phase2_search.tar.gz` | 2.0 MiB | 2.1 MiB | Cached ParEGO episodes for three starts |
| `phase3_campaigns.tar.gz` | 6.6 MiB | 78.0 MiB | Offline rare-event campaign replay |
| `sim01_0.9.14-dirty.tar.gz` | 768.6 MiB | 1469.6 MiB | Custom CARLA map for intersection2 |

The sizes are file payload sizes; allow additional space for filesystem metadata,
training outputs and simulator logs. The custom map needs about 1.44 GiB of new
CARLA content, in addition to its approximately 769 MiB archive. Check the target
filesystem before extracting; the simulator and its dependencies need their own
space.

## Check the downloads

Place the chosen data archives in the repository root. Verify them against the
checksums shipped with this source version:

```bash
sha256sum --ignore-missing -c release-assets.sha256
```

This checks each listed archive present in the current directory; missing
optional archives are skipped. Every selected archive must report `OK` before
it is extracted. The release also includes `SHA256SUMS`, which additionally covers
`simreq-source.tar.gz`; use the same command with that filename in the download
directory to verify the source archive.

## Training and audit evidence

```bash
mkdir -p episode_dataset
tar -xzf phase1_labelling.tar.gz -C episode_dataset --strip-components=1
# Optional historical audit evidence, in its own directory:
tar -xzf phase1_audit.tar.gz
```

Train from `episode_dataset/`. The second command creates `phase1_audit/`.
Keep it separate: both archives use `episodes.000001.parquet` filenames, so
extracting both into `episode_dataset/` would overwrite the training shards.
The historical audit archive is for inspection; a new audit runs its own frozen
sampling plan and writes fresh evidence.

## Search episodes and campaign replay

```bash
tar -xzf phase2_search.tar.gz
tar -xzf phase3_campaigns.tar.gz
```

Each archive includes its own README. ParEGO uses matching design-and-seed
cache entries from the selected start; cache misses require simulation. Campaign
replay recomputes the published endpoints entirely offline. See the main
[workflow](README.md) for training, search and replay commands.

## Custom CARLA map

Set `CARLA_ROOT` to the directory containing `CarlaUE4.sh` and `ImportAssets.sh`.
Run these commands from the repository root after downloading and checking the
map archive:

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.14
df -h "$CARLA_ROOT"
mkdir -p "$CARLA_ROOT/Import"
cp sim01_0.9.14-dirty.tar.gz "$CARLA_ROOT/Import/"
(cd "$CARLA_ROOT" && ./ImportAssets.sh)
```

The importer unpacks archives in `Import/`, including any other packages already
there. The package installs under `CarlaUE4/Content/sim01/`; scenario
`intersection2` selects the `sim01` map. `intersection1` uses the standard
`Town05` map. The packaged map and OpenDRIVE file match those used for the live
CARLA smoke check. Archive ownership metadata is normalized; the map file
contents are unchanged.
