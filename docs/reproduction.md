# Reproducing the paper

[← Project overview](../README.md)

## Run the paper

Run all commands from the repository root. You need Python ≥ 3.11, a C++17 compiler,
Docker, GNU Make, and GNU Parallel. 

```sh
uv sync
source .venv/bin/activate
make baselines             # build the core, FiVI, and SAYNT images once
make reproduce JOBS=8      # run all paper experiments; resume completed jobs
make figures              # plot saved results; no solver runs
```

The full run takes tens of CPU hours, plus the extended navigation runs. SAYNT's
image compiles Storm and can take a while to build. Budget details are listed below.

`make reproduce` runs the main, main-extended, navigation, navigation-extended, and
Tiger plans, followed by the distractor study. Plans run in sequence and stop on error.
Rerun the same command to resume; completed jobs are skipped. `JOBS` defaults to 1
and controls concurrency in the two main benchmark plans. Both navigation plans run
one job at a time because FiVI requests 48 GiB heaps; extended 20×20 navigation uses
32 threads. Tiger and the distractor study also run sequentially.

To use another result location or Python interpreter:

```sh
make reproduce JOBS=4 OUTPUTS=/path/to/results
make figures OUTPUTS=/path/to/results
# Alternatively, without activating the virtual environment:
make figures PYTHON=.venv/bin/python
```

Result directories are `OUTPUTS/main-parallel/job-N/`,
`OUTPUTS/main-extended-parallel/job-N/`, `OUTPUTS/navigation-parallel/job-N/`,
`OUTPUTS/navigation-extended-parallel/job-N/`, `OUTPUTS/tiger/`, and
`OUTPUTS/distractor/`. `make figures` expects that layout and checks that these
directories exist before plotting. It writes the main, navigation, and Tiger outputs
to `OUTPUTS/figures/`; distractor plots stay in `OUTPUTS/distractor/`.
With saved results in this layout, only the local package and Make are needed to plot;
Docker and GNU Parallel are needed to run experiments.

Run `make` or `make help` to list the targets. The individual commands below remain
available if you only need one experiment or figure.

For a fixed source revision, check it out before building the images. Record the
checkout revision and image IDs when comparing results; the `latest` registry tag can change.

## Quick start with Docker

The image contains the package, all models and the experiment plans. It needs no network
access at runtime.

```sh
docker pull ghcr.io/oiver12/predictive-fsc:latest
mkdir -p outputs
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$PWD/outputs:/outputs" ghcr.io/oiver12/predictive-fsc \
  solve models/tiger_episodic.toml --memory 4 --cpu-seconds 30 --output /outputs/tiger.npz
docker run --rm -v "$PWD/outputs:/outputs" ghcr.io/oiver12/predictive-fsc \
  evaluate models/tiger_episodic.toml /outputs/tiger.npz
```

To build the image yourself:

```sh
docker build --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t predictive-fsc .
```

The image runs MAPO and the random-initialization baseline (`solve`, `halving`, `evaluate`,
`figures`, and plans without external solvers). FiVI and SAYNT run in their own containers,
which the host-installed CLI starts (see [Baselines](#baselines)).

## Local installation

Requires Python ≥ 3.11 and a C++17 compiler. The EM fit is a C++ extension built during installation.

```sh
uv sync                        # or: pip install -e .
source .venv/bin/activate
predictive-fsc validate models/*.toml
predictive-fsc solve models/tiger_episodic.toml --memory 4 --cpu-seconds 30
predictive-fsc halving models/hallway.toml --memory 2 4 8 16 --cpu-seconds 150
```

`solve` runs one search; `halving` runs successive halving over several memory sizes. Every
run is single-threaded and saves an `.npz` with the controller, the candidate trace, settings,
seeds and build provenance. `predictive-fsc <command> --help` lists all options. A search stops at
`--cpu-seconds` or `--rounds`, whichever comes first.

## Reproducing the paper

Each experiment is a TOML plan of explicit `[[runs]]` entries in `experiments/`:

| Plan | Jobs | Contents |
| --- | ---: | --- |
| `main.toml` | 308 | 22 benchmarks × 4 seeds: predictive and random halving over M = 2/4/8/16, random M = 1, FiVI and SAYNT at matched budgets |
| `main-extended.toml` | 22 | FiVI references, 7,200 CPU seconds |
| `navigation.toml` | 39 | 3 layouts × 4 seeds: predictive M = 4, random M = 4 and M = 1, FiVI |
| `navigation-extended.toml` | 3 | FiVI references: W20 7,200 wall seconds on 32 threads, W16 7,200 CPU seconds; 48 GiB heap |
| `tiger.toml` | 2 | Tiger H = 8, M = 4: 2,000 random starts and a FiVI reference |
| `appendix/distractor.py` | 240 | Distractor T-maze: 8 distractor lengths × 30 repetitions, M = 2 |

Budgets: B = 150 CPU seconds per memory size (600 for hallway, mit and both RockSample
problems). Halving uses 3B in total, and the external baselines get the same 3B. Navigation runs get 600 CPU seconds.

### Running plans

```sh
predictive-fsc reproduce experiments/main.toml --dry-run          # list all jobs
predictive-fsc reproduce experiments/main.toml --output outputs/main
predictive-fsc reproduce experiments/main.toml --run-index 7 --output outputs/main   # one job (1-based)
```

Completed jobs are skipped when a plan is rerun; `--force` reruns them.

### In parallel

Every job is independent and single-threaded. `tools/reproduce_parallel.py` checks the whole
plan, then uses [GNU Parallel](https://www.gnu.org/software/parallel/) to run one
`reproduce --run-index N` process per job, at most `--jobs` at a time:

```sh
python tools/reproduce_parallel.py experiments/main.toml --jobs 16 --output outputs/main-parallel
python tools/reproduce_parallel.py experiments/main-extended.toml --jobs 16 --output outputs/main-extended-parallel
python tools/reproduce_parallel.py experiments/navigation.toml --jobs 1 --output outputs/navigation-parallel
python tools/reproduce_parallel.py experiments/navigation-extended.toml --jobs 1 --output outputs/navigation-extended-parallel
```

Each job writes to `job-N/` (results, stdout, stderr), and `joblog.tsv` records timing and exit
status. Rerunning the same command resumes the batch and skips finished jobs. Ctrl-C stops the batch and removes
its solver containers. Choose `--jobs` from your core count and memory. Navigation FiVI jobs
need large heaps, so run them one at a time.

### Figures and tables

```sh
predictive-fsc figures outputs/main-parallel/job-*/ outputs/main-extended-parallel/job-*/ --paper main --output outputs/figures/main
predictive-fsc figures outputs/navigation-parallel/job-*/ outputs/navigation-extended-parallel/job-*/ --paper navigation --output outputs/figures/navigation
predictive-fsc figures outputs/tiger --paper tiger --output outputs/figures/tiger
python experiments/appendix/distractor.py --output outputs/distractor           # run
python experiments/appendix/distractor.py --output outputs/distractor --plot    # plot
```

| Paper item | Output |
| --- | --- |
| Figure 1a | `--paper tiger` |
| Figure 3, Table 1, Figure 4 | `--paper main` (initialization panels, `returns-and-storage.csv`, per-benchmark anytime) |
| Table 2, Figure 5 | `--paper navigation` |
| Figure 6 / Appendix C.1 | `appendix/distractor.py --plot` |

Plotting reads saved results only and never runs a solver.

## Baselines

FiVI and SAYNT run in their own pinned Docker images. `make baselines` builds them
and the core image in order. The equivalent commands are:

```sh
docker build -t predictive-fsc .                                      # FiVI builds on the core image
docker build --platform linux/amd64 -t predictive-fsc-fivi baselines/fivi
docker build -t predictive-fsc-saynt baselines/saynt                  # compiles Storm; slow
```

- **FiVI** builds [our corrected fork](https://github.com/oiver12/ConstrainedPlanningToolbox)
  of Walraven and Spaan's ConstrainedPlanningToolbox at a pinned revision, from a checksummed
  archive. Reported values are lower bounds, checked
  against the initial alpha vector. `FIVI_HEAP` and `FIVI_PROCESSORS` set the resources.
- **SAYNT** builds pinned [PAYNT](https://github.com/randriu/paynt),
  [Storm](https://github.com/randriu/storm), carl, pycarl and
  [Stormpy](https://github.com/oiver12/stormpy) (our fork with an ownership fix) and runs PAYNT
  directly, with 60 s and 10 s phases. Finite horizons are encoded by unrolling stages and
  shifting rewards.

The adapters (`src/predictive_fsc/baselines/`) convert each problem, run one container per job
without network access and record the image ID with the result. Upstream sources and licenses
are kept in the images under `/opt/fivi/source` and `/opt/saynt/source`.

## Problems

Each problem is a standard [Cassandra `.pomdp`](https://pomdp.org/code/pomdp-file-spec.html)
model plus a small TOML description:

```toml
model = "tiger_episodic.pomdp"
horizon = 8
discount = 1.0
terminal_actions = ["open_left", "open_right"]
terminal_states = []
```

All five fields are required. The objective is the undiscounted finite-horizon return.
Termination is never inferred, so terminal actions and states must be listed explicitly. A
terminal step still receives its reward and observation. Rewards are reduced to the expected
`r(s, a)`. To add a problem, write a `.pomdp` file and a TOML file next to it, then run
`predictive-fsc validate`.

`models/` contains the 22 benchmarks, three navigation layouts and eight distractor T-mazes.
Many benchmarks come from Tony Cassandra's [POMDP examples](https://pomdp.org/examples/).
Per-model source attribution for the custom variants is still being completed.
New RockSample and navigation variants can be generated from explicit parameters:

```sh
python tools/generate_rocksample.py --help
python tools/generate_navigation.py --help
```
