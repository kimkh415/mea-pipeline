# HPC scripts (FAS RC)

How to install, configure and run: the [main README](../../README.md).
This page is what the scripts do.

## The files

| | what it is | where it runs |
|---|---|---|
| `01_setup_env.sh` | finds or installs conda, builds the env (`MEA_CONDA_ENV`, recorded in `.mea_env`), finds or clones the repo | `gpu_test`, once, interactively |
| `submit_samples.sh` | reads `samples.tsv` and `mea.env`, checks every line, calls `submit_recording.sh` per line | login node |
| `submit_recording.sh` | submits one recording: `02` then `03` | login node |
| `02_sort.sh` | Kilosort 4 + bombcell on one recording, via `run_kilosort4.py` | `gpu`, one A100 80 GB |
| `03_downstream.sh` | `mea run` (QC, metrics, figures) on the sorted output, then `05_archive.sh` | `shared` |
| `05_archive.sh` | copies one recording from scratch to `MEA_FINAL`; also run by hand after a killed downstream job | `shared` |
| `_common.sh` | paths, repo and env lookup; sourced by the others | — |

Setup runs on a GPU node because pip picks the CUDA or CPU build of torch from
the driver it sees; a login node has none, and a CPU torch fails only 40
minutes into a sort. Setup asserts `torch.cuda.is_available()` and refuses to
run without a GPU.

## Where things go

```
$MEA_SCRATCH/sorted/<prefix>/    sorter output; recording.dat (~476 GB) while sorting
$MEA_SCRATCH/results/<prefix>/   mea run output
$MEA_FINAL/<prefix>/             figures/, compiled_metrics/, pipeline/ (.mea store, sort/<prefix>/, logs/)
<project>/logs/                  every job's .out/.err
```

Netscratch is not backed up and is purged on a retention window. The archive
leaves the work directory in place (`MEA_ARCHIVE_PURGE=1` deletes it), so it
can be re-run.

## Job chain

```
sort_<prefix>  (gpu)  --afterok-->  downstream_<prefix>  (shared): mea run, then archive
```

- A sort clears any earlier attempt at the same prefix; `MEA_REUSE_SORT=1`
  relabels the existing sort instead, on a CPU node.
- A failed sort cancels the downstream job.
- The archive runs even when `mea run` fails, and the job keeps `mea run`'s
  exit status. A job killed for time or memory skips it: `sbatch 05_archive.sh
  <prefix>` from the project folder.
- `MEA_MAIL`: a mail when the downstream job ends, and when a sort fails.

## If it fails

| | |
|---|---|
| sort sits idle after "Re-computing universal templates" | OpenBLAS thread-pool deadlock; `run_kilosort4.py` sets `OPENBLAS_NUM_THREADS=1` before any import |
| GPU too small | the driver checks the card and stops before writing the binary |
| out of scratch | `02_sort.sh` refuses at the start |
| no conda env | `submit_recording.sh` refuses on the login node |
| sort OOM-killed | `MEA_SORT_MEM=400G` |
