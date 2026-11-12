.DEFAULT_GOAL := help

PYTHON ?= python
JOBS ?= 1
OUTPUTS ?= outputs

.PHONY: help baselines reproduce figures

help:
	@printf '%s\n' \
	  'Paper reproduction (run from the repository root):' \
	  '  make baselines          Build the core, FiVI, and SAYNT Docker images' \
	  '  make reproduce JOBS=8   Run all paper experiments; skip completed jobs' \
	  '  make figures            Generate figures and CSV tables from saved results' \
	  '' \
	  'Setup: uv sync && source .venv/bin/activate' \
	  'Experiments also require Docker and GNU Parallel.' \
	  'Defaults: JOBS=1, OUTPUTS=outputs, PYTHON=python' \
	  'Navigation plans run one job at a time (48 GiB FiVI heaps).' \
	  'Results: OUTPUTS/<plan>-parallel/job-N/, OUTPUTS/tiger/, OUTPUTS/distractor/' \
	  'Figures: OUTPUTS/figures/{main,navigation,tiger}; OUTPUTS/distractor/' \
	  'See docs/reproduction.md for budgets and individual commands.'

baselines:
	docker build --build-arg SOURCE_REVISION="$$(git rev-parse HEAD)" -t predictive-fsc .
	docker build --platform linux/amd64 -t predictive-fsc-fivi baselines/fivi
	docker build -t predictive-fsc-saynt baselines/saynt

reproduce:
	$(PYTHON) tools/reproduce_parallel.py experiments/main.toml --jobs $(JOBS) --output "$(OUTPUTS)/main-parallel"
	$(PYTHON) tools/reproduce_parallel.py experiments/main-extended.toml --jobs $(JOBS) --output "$(OUTPUTS)/main-extended-parallel"
	$(PYTHON) tools/reproduce_parallel.py experiments/navigation.toml --jobs 1 --output "$(OUTPUTS)/navigation-parallel"
	$(PYTHON) tools/reproduce_parallel.py experiments/navigation-extended.toml --jobs 1 --output "$(OUTPUTS)/navigation-extended-parallel"
	$(PYTHON) -m predictive_fsc reproduce experiments/tiger.toml --output "$(OUTPUTS)/tiger"
	$(PYTHON) experiments/appendix/distractor.py --output "$(OUTPUTS)/distractor"

figures:
	@for directory in main-parallel main-extended-parallel navigation-parallel navigation-extended-parallel tiger distractor; do \
	  test -d "$(OUTPUTS)/$$directory" || { printf 'Missing results: %s\nRun make reproduce or extract saved results here first.\n' "$(OUTPUTS)/$$directory" >&2; exit 1; }; \
	done
	$(PYTHON) -m predictive_fsc figures "$(OUTPUTS)"/main-parallel/job-*/ "$(OUTPUTS)"/main-extended-parallel/job-*/ --paper main --output "$(OUTPUTS)/figures/main"
	$(PYTHON) -m predictive_fsc figures "$(OUTPUTS)"/navigation-parallel/job-*/ "$(OUTPUTS)"/navigation-extended-parallel/job-*/ --paper navigation --output "$(OUTPUTS)/figures/navigation"
	$(PYTHON) -m predictive_fsc figures "$(OUTPUTS)/tiger" --paper tiger --output "$(OUTPUTS)/figures/tiger"
	$(PYTHON) experiments/appendix/distractor.py --output "$(OUTPUTS)/distractor" --plot
