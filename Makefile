PY := .venv/bin/python
UV := uv

.DEFAULT_GOAL := help
.PHONY: help setup test lint verify toy data score analyse paper reproduce clean

help:  ## Show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install the package
	$(UV) venv --python 3.11
	$(UV) pip install --python .venv/bin/python -e ".[dev]"

test:  ## Verify Theorems 1-5 and Prop 7 numerically (CPU, ~1 min, no downloads)
	$(PY) -m pytest tests/ -q

lint:  ## Lint and format-check
	.venv/bin/ruff check src tests

verify: test toy  ## Full offline verification: assertions + measured claim ledger

toy:  ## Machine-check all 24 claims on synthetic ground truth -> ledger + figures
	$(PY) -m scripts.toy --out results/runs/toy --seeds 0,1,2

data:  ## Download pinned benchmark revisions
	$(PY) -m scripts.download --config configs/benchmarks

score:  ## GPU stage: response tensors for the model grid (~20 GPU-h on one 3090)
	$(PY) -m scripts.score --config configs/experiments/core.yaml

analyse:  ## IRT fits, identified sets, DIF tests with BH-FDR
	$(PY) -m scripts.analyse --config configs/experiments/core.yaml

paper:  ## Regenerate every table and figure, then build the PDF
	$(PY) -m scripts.make_tables
	$(PY) -m scripts.make_figures
	cd paper && latexmk -pdf main.tex

reproduce: test toy data score analyse paper  ## Everything, in order

clean:
	rm -rf results/runs/* paper/figures/*.pdf paper/tables/*.tex
	find . -name __pycache__ -type d -exec rm -rf {} +
