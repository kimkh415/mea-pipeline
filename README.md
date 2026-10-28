# mea-pipeline

Spike sorting, unit QC and network metrics for 3Brain HD-MEA recordings of
organoids under a drug series ending in TTX.

1. **Join:** an organoid's `.brw` files, one per condition (baseline, drugs,
   TTX), are put end to end in acquisition order into one full recording, so
   each unit is sorted once and keeps one identity across conditions.
2. **Sort** (GPU): **[Kilosort 4](https://github.com/MouseLand/Kilosort)** on
   all 4096 electrodes in one pass, drift correction on;
   **[bombcell](https://github.com/Julie-Fabre/bombcell)** labels each cluster.
3. **Unit QC:** drops units with > 5 % of spikes within 1 ms of each other,
   units below 0.03 Hz in every condition, units TTX does not silence (< 90 %
   drop from peak rate), one of each pair firing in lockstep (crosstalk), and
   units below 0.04 Hz in every non-TTX condition. Keeps bombcell types 1–4 or
   Kilosort `good`.
4. **Metrics:** per condition, on its last 450 s (the whole condition if
   shorter): firing, synchrony, single-unit bursting, network bursts
   ([Metrics](#metrics)).
5. **Output:** per recording, 10 diagnostic PDFs and a metrics table; across
   recordings, one table.

**Contents:** [Setup](#setup-on-the-hpc) · [Configure](#configure-a-project) · [Run](#run) · [Output](#output) · [Metrics](#metrics) · [Analysis only](#analysis-only-no-sorting) · [Common problems](#common-problems)

## Setup on the HPC

Once, on FAS RC.

1. Save `mea-pipeline-v1.1.0.tar.gz` in the directory where you want the
   pipeline, and extract it:

   ```bash
   tar zxf mea-pipeline-v1.1.0.tar.gz          # creates mea-pipeline/
   ```

2. Get a GPU session:

   ```bash
   salloc -p gpu_test --gpus 1 -c 8 --mem 32G -t 0-04:00
   ```

3. In that session, build the environment (~20 min), then leave:

   ```bash
   bash mea-pipeline/scripts/slurm/01_setup_env.sh   # ends with "environment ready"
   exit
   ```

   This builds a conda environment `MEA` with Kilosort 4, bombcell, a CUDA
   torch and the pipeline. It uses the cluster's conda, available via
   `module load Mambaforge`.

> [!NOTE]
> The environment is ~10 GB and lands in your home directory
> (`~/.conda/envs/MEA`), which has a small quota. To put it on lab storage,
> set these before step 3:
>
> ```bash
> export MEA_CONDA_ENV=/path/on/lab/storage/conda/MEA       # the environment
> export CONDA_PKGS_DIRS=/path/on/lab/storage/conda/pkgs    # conda's download cache
> export PIP_CACHE_DIR=/path/on/lab/storage/pip-cache       # pip's download cache
> ```
>
> Setup remembers the environment's location; the jobs find it with nothing
> else set.

## Configure a project

A project folder is where you submit from. **It holds two files you write:**

**1. `samples.tsv`**: one recording per line, *tab*-separated: the folder
holding its `.brw` files, and the filename prefix they share.

```
directory	prefix
/path/to/recording_directory	392_het_100Hz
```

A recording is one `.brw` per condition, named prefix + drug; the baseline has
no drug:

```
392_het_100Hz_00.brw                  baseline
392_het_100Hz_gabazine_20uM_00.brw    gabazine
392_het_100Hz_TTX_2uM_00.brw          TTX (required)
```

Drug names come from a list, `dataset.conditions.labels` in
[`config/default.yaml`](config/default.yaml): `Gabazine`, `CGP`, `DNQX`,
`D-AP5`, `TTX`. A file takes a drug's label when that name appears in the
filename (case-insensitive); the file with none is the baseline. A file
matching no name keeps its filename as the condition, and the run warns. So
`D_AP5` (underscore) is not recognised as `D-AP5`: rename the file, or add the
spelling or a new drug to the list. Conditions are ordered by the acquisition
time stored in each file.

**2. `mea.env`**: where runs go.

```bash
MEA_SCRATCH=/n/netscratch/arlotta_lab/Users/$USER/mea   # work directory, 100s of GB per running sort
MEA_FINAL=/path/on/lab/storage/results                  # kept results
MEA_MAIL=you@example.org                                # optional: a mail when each recording is done
```

`MEA_SCRATCH` and `MEA_FINAL` are required. Instead of `mea.env` they can be
exported in your shell; `mea.env` wins over the shell, and `--scratch DIR` /
`--final DIR` on the command line win over both.

<details>
<summary>Other settings for <code>mea.env</code></summary>

| Variable | Default | Controls |
|---|---|---|
| `MEA_REUSE_SORT` | – | `1`: keep the existing sort and only redo the steps after it (default: clear it and sort again) |
| `MEA_MIN_BRW` | 2 | `.brw` files a recording needs to be submitted; `1` for single-condition runs |
| `MEA_SORT_MEM` | 250G | Memory for the sort job |
| `MEA_KEEP_DAT` | – | `1`: keep the sort's raw binary (100s of GB) |
| `MEA_ARCHIVE_PURGE` | – | `1`: delete a sort from scratch once it is copied to `MEA_FINAL` |
| `MEA_CONDA_ENV` | `MEA`, or what setup recorded | Conda environment: a name, or a path |
| `MEA_CONSTRAINT` | `a100` | GPU type the sort asks Slurm for (`--constraint`). The sort needs ~65 GB of GPU memory, and on FAS RC only the A100 nodes have 80 GB cards |

</details>

Analysis settings (QC thresholds, metric windows, drug names) are in
[`config/default.yaml`](config/default.yaml), with a comment on each.

## Run

From a login node, in the project folder:

```bash
bash /path/to/mea-pipeline/scripts/slurm/submit_samples.sh samples.tsv --dry-run   # every line must be ok
bash /path/to/mea-pipeline/scripts/slurm/submit_samples.sh samples.tsv
squeue -u $USER -o "%.10i %.9P %.35j %.8T %.10M %R"                                # to check status
```

Each recording is two jobs:

| Job | Runs on | Takes | Does |
|---|---|---|---|
| `sort_<prefix>` | GPU | ~3 h | Kilosort 4 + bombcell |
| `downstream_<prefix>` | CPU | ~30 min | QC, metrics, figures, then copies the results to `MEA_FINAL/<prefix>` |

Logs are in `logs/`. With `MEA_MAIL` set you get a mail when
`downstream_<prefix>` ends, and one if a sort fails.

**Failed jobs:**

```bash
sacct -u $USER -S 2026-10-20 --state=FAILED,TIMEOUT,OUT_OF_MEMORY   # since a date (YYYY-MM-DD); or -S now-2days
```

Put their lines in a new list and submit it; a resubmitted sort clears the
failed attempt and starts over. If only `downstream_<prefix>` failed, submit
with `MEA_REUSE_SORT=1` to skip the 3 h sort. If it was killed for time or
memory, its results were not copied: run
`sbatch /path/to/mea-pipeline/scripts/slurm/05_archive.sh <prefix>` from the
project folder.

## Output

Per recording, in `MEA_FINAL/<prefix>/`:

```
figures/                    10 diagnostic PDFs
compiled_metrics/
  compiled.wide.csv         metrics: one row per condition, one column per metric
pipeline/
  <prefix>.mea/             the recording's store:
    units.parquet             kept units: id, channel, x, y, labels, TTX drop
    spikes.parquet            spike times (ms) per unit
    conditions.parquet        condition names, start/end (s), is_ttx
    qc.parquet                every QC decision: unit, step, reason
    artifacts/                per-step arrays
  sort/<prefix>/            Kilosort + bombcell output; kilosort4_0.tar.gz is the
                            full sorter folder without the raw binary
  logs/                     this recording's job logs
```

QC deletes nothing: dropped units stay in the store, marked, with the reason
in `qc.parquet`.

**All recordings in one table**, with the `MEA` environment active
(`conda activate MEA`):

```bash
mea compile --input $MEA_FINAL --out compiled --samples sample_info.csv
```

writes `compiled/compiled.wide.csv`: one row per recording × condition.
`--samples` is optional: a CSV with a `recording` column (the prefix) and any
columns you want next to it, which you can then filter and group on:

```
recording,genotype,line,batch
392_het_100Hz,het,ARID1B,5
384_ctr_100Hz,ctr,ARID1B,5
```

<details>
<summary>Figures</summary>

| File | Check |
|---|---|
| `unit_QC` | Units left after each QC step |
| `ttx` | Every unit's peak vs TTX rate; kept units drop ≥ 90 % |
| `waveforms` | Mean waveforms by bombcell class and by QC verdict; kept units should be spike-shaped |
| `isi` | Refractory violations vs firing rate |
| `raster` | Kept units over the whole recording, conditions shaded |
| `overview` | Raster and population rate end to end, condition boundaries marked |
| `activity_per_condition` | Population trace per condition with detected network events; marks should sit on the peaks |
| `metrics` | Main metrics across conditions |
| `electrode_map` | Firing density on the array per condition |
| `electrode_scatter` | Kept units on the array, sized by rate |

</details>

## Metrics

One row per recording × condition. Unit values are means over the kept units;
`cv` = SD / mean; blank = not measured (e.g. no network bursting).

Every metric is measured on the last 450 s of its condition, or the whole
condition if shorter (`experiment.window_s`; the span is in `t_start_s`,
`t_end_s`). `n_units` is the exception: all units kept by QC, which uses whole
conditions. The 450 s comes from the original MATLAB
pipeline (`Analyze_spikeTrain.m`), which gives no reason for it. What each
metric tells you, in plain words: [METRICS.md](METRICS.md).

| Row info | |
|---|---|
| `recording`, `condition`, `treatment` | Recording (prefix), its condition file, and the drug label |
| `t_start_s`, `t_end_s` | Analysis window within the joined recording (s) |
| `n_units` | Units analysed |

**Firing** (per unit, averaged)

| Metric | Meaning |
|---|---|
| `firingRate` | Firing rate (Hz); = `firingRateCount` |
| `firingRateISI` | 1 / mean interval between spikes (Hz): the rate while the unit is firing. Pulled up by units that fire only a few, close spikes |
| `firingRateCount` | Spikes / window length (Hz): includes silent stretches |
| `fanoFactor` | Variance / mean of spike counts in fixed 100 ms windows. 1 = Poisson; < 1 more regular; > 1 spikes clump (bursts). A fixed window makes it depend on rate: at ~1 Hz most bins hold 0 or 1 spike, which gives ≈ 1 for any unit. Prefer `fanoFactorOp` when comparing conditions |
| `fanoFactorOp` | The same in operational time (Rajdl, Lansky & Kostal 2020): each unit's window is 5 of its own mean intervals (`fanoOpWindow_isi`), so it does not move with firing rate. A unit is included if it has ≥ 2 spikes and its mean interval is ≤ 1/50 of the analysis window (9 s for 450 s, 6 s for 300 s), so the window fits ≥ 10 times |
| `fanoOpUnits` | Units included in `fanoFactorOp` by that rule; the rest of `n_units` fire too slowly to qualify |
| `fanoFactorCV2` | Squared CV of the intervals between spikes; another variability measure, which changes with rate |
| `lvr` | Local variation (Lv): how much each interval differs from the next. < 1 regular, ≈ 1 random, > 1 bursty (Shinomoto et al. 2003) |
| `entropy` | Shannon entropy (bits) of the distribution of a unit's spike counts per 30 ms bin |

**Synchrony** (unit pairs, averaged)

| Metric | Meaning |
|---|---|
| `sttc` | Spike-time tiling coefficient (Cutts & Eglen 2014) at ±100 ms, as in Negri et al. 2020: do two units fire within 100 ms of each other more often than their rates alone would give? 0 = no more than chance; 1 = every spike of each has a partner; negative = they fire together less than chance |
| `sttc_dt15`, `sttc_dt100` | STTC at ±15 / ±100 ms |
| `sttc_dt15_abs`, `sttc_dt100_abs` | Mean of \|STTC\| |
| `sttcNegativeFraction` | Fraction of pairs with negative STTC |
| `mutualInfo` | Mutual information (bits) between two units' spike counts in the same 30 ms bins, from their joint count distribution; 0 = independent |

**Single-unit bursting**: bursts are found per unit with the Maximum Interval method (as in meaRtools, Gelfman et al. 2018); units with ≥ 5 bursts are *bursting*, the rest *tonic*.

| Metric | Meaning |
|---|---|
| `n_bursting`, `n_tonic` | Number of bursting / tonic units |
| `burst.neurons`, `tonic.neurons` | Fraction of units bursting / tonic |
| `burst.rate_mean`, `_cv` | Bursts per minute |
| `burst.duration_mean`, `_cv` | Burst length (s) |
| `burst.amplitude_mean`, `_cv` | Firing rate inside bursts (Hz) |
| `burst.nSpikes_norm` | Fraction of a unit's spikes in bursts |
| `burst.rate_ff`, `burst.amplitude_ff` | CV² of the gaps between bursts / of the intervals within bursts |
| `nonburst.spikeRate_mean`, `_cv`, `_ff` | Firing rate outside bursts in bursting units (Hz) |
| `tonic.spikeRate_mean`, `_cv`, `_ff` | Firing rate of tonic units (Hz) |

**Network bursting**: an event = ≥ 20 % of units active within 100 ms (events < 1 s apart merged).

| Metric | Meaning |
|---|---|
| `network.nBurst` | Network events. Blank = condition not network bursting; < 3 = the network values below rest on few events |
| `network.n_channels` | Units that significantly take part in events |
| `network.background_fraction` | Median fraction of units active outside events |
| `network.background_flag` | 1 = background high enough that chance overlap could pass as events |

**Network burst families**: events grouped into families of similar events, one block per family (`family1.`, `family2.`, …).

| Metric | Meaning |
|---|---|
| `families.n_cluster` | Number of families |
| `families.quality` | 2 = best grouping significant, 1 = a smaller one is, 0 = one family |
| `familyN.nBurst` / `.nBurstTotal` | Events in this family / in the condition |
| `familyN.rate.mean`, `.cv` | Events per minute |
| `familyN.duration.mean`, `.cv` | Event length (s) |
| `familyN.amplitude.mean`, `.cv` | Firing rate of participating units within an event (Hz) |
| `familyN.height.mean`, `.cv` | Event peak normalised by its spikes and duration (a.u.) |
| `familyN.nSpikes.mean` | Spikes per participating unit per event |
| `familyN.nSpikes.norm`, `.normTotal` | Share of participants' / of all spikes in these events |
| `familyN.neuronsAct.mean`, `.cv` | Fraction of participating units active in an event |
| `familyN.delay.mean`, `.cv` | Latency of each unit's first spike after event onset (ms) |
| `familyN.distance.mean`, `.cv` | Spread of units' spikes around the event peak (ms) |
| `familyN.nNeuron.totalBurst`, `.normRec` | Units in network events (count, fraction of all units) |
| `familyN.nNeuron.Burst`, `.normBurst` | Units in this family (count, fraction of participants) |

## Analysis only (no sorting)

To rerun QC and metrics, e.g. with other settings, from an archived sort
(`MEA_FINAL/<prefix>/pipeline/sort/<prefix>`) or any folder with
`spike_times.npy`, `spike_clusters.npy`, `metadata.csv`, `recording_data.csv`:

```bash
tar zxf mea-pipeline-v1.1.0.tar.gz
conda create -n MEA python=3.12 pip && conda activate MEA
pip install -e mea-pipeline

mea plan --input SORTED                                    # what would run
mea run  --input SORTED --out RESULTS                      # QC, metrics, figures, table
mea run  --input SORTED --out RESULTS --set experiment.window_s=300 --set qc_manual.min_activity_hz=0.05
mea run  --input SORTED --out RESULTS --config my.yaml     # my.yaml: only the keys you change
mea compile --input RESULTS --out RESULTS/compiled
```

Before sorting, from the `.brw` files:

```bash
mea conditions --input FOLDER    # each recording's files, condition order and drug labels
mea rawqc --input FOLDER         # does TTX silence enough electrodes to be worth sorting?
```

`mea rawqc` uses no sorting: for each electrode it reads, from the file
headers, the fraction of time the recorder saw it above its noise threshold
(~15 s per recording). An electrode is *active* above twice the chip's median,
and *TTX-sensitive* if it is active at baseline and TTX halves it or more. A
recording passes with > 30 TTX-sensitive electrodes.

## Common problems

| Message | Fix |
|---|---|
| `not set: MEA_SCRATCH MEA_FINAL` | Write `mea.env` in the project folder |
| `ERROR: ... listed twice` | Remove the duplicate line; each prefix is one recording |
| `need a baseline plus TTX` | Fewer than 2 `.brw` files match the prefix; check the folder and prefix |
| `these conditions match no label` | Rename the file, or add the spelling to `dataset.conditions.labels` |
| `files carry no acquisition timestamp` | Set `dataset.discovery.condition_order: explicit` and `order` |
| Sort job `OUT_OF_MEMORY` | `MEA_SORT_MEM=400G` in `mea.env`, resubmit |
