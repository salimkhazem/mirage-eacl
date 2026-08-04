# MIRAGE

## The Cross-Lingual Ability Gap Is Not Identified

Multilingual evaluation routinely reports that a model “loses _X_ points” when moving from a source language to a target language. That observed score drop is
real, but its usual interpretation is not automatic: it combines a change in model ability with any change in item difficulty introduced by translation.

MIRAGE formalizes this ambiguity. It shows that the cross-lingual _ability_ gap is not identified from a parallel benchmark—even when every model answers every item in every language. The repository provides the theory, estimators, simulation checks, empirical analyses, robustness studies, and manuscript
artifacts supporting that result.

## What the data can and cannot identify

| Quantity                                     |        Rasch model |           Language-varying 2PL |
| -------------------------------------------- | -----------------: | -----------------------------: |
| Within-language model ranking                |         Identified |                     Identified |
| Magnitude of within-language model contrasts |         Identified |    Identified only up to scale |
| Difference-in-differences across models      |         Identified |      Not identified in general |
| Relative drift between items                 |         Identified | Not generally point-identified |
| Level of the cross-lingual ability gap       | **Not identified** |             **Not identified** |
| Uniform translation drift                    | **Not identified** |             **Not identified** |

The practical message is deliberately narrow: per-language rankings remain meaningful under the stated model, but a reported cross-lingual gap should not
be interpreted as pure ability loss without an explicit translation-invariance assumption or a corresponding sensitivity analysis.

## Empirical study

The analysis covers eight parallel multilingual benchmarks, seven multiple-choice tasks and one generative task. The response panel contains 19 open-weight models spanning 1.2B–9.2B parameters and 152 language gaps.

| Result                                                   | Committed evidence |
| -------------------------------------------------------- | -----------------: |
| Parallel benchmarks                                      |                  8 |
| Evaluated models                                         |                 19 |
| Language gaps                                            |                152 |
| Item–language cells tested for non-uniform drift         |            212,310 |
| Detected drift range across benchmarks, with FDR control |         2.6%–25.7% |
| Gaps below one observed-drift standard deviation         |       93/152 (61%) |
| Degeneracy-robust count                                  |       91/152 (60%) |

## Quick verification

The theory and synthetic claim ledger can be checked on CPU without model weights, benchmark downloads, or a GPU.

Requirements: Python 3.10 or later and `uv`.

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m scripts.toy --out results/runs/toy --seeds 0,1,2
```

## Full empirical reproduction

The full scoring stage requires a CUDA-capable environment and access to the configured public datasets and model checkpoints.
Install the additional dependencies first:

```bash
uv pip install --python .venv/bin/python -e ".[gpu]"
```

Then run the stages in order:

```bash
# Validate and download the configured benchmark revisions.
.venv/bin/python -m scripts.download --config configs/benchmarks.yaml

# Score multiple-choice tasks. Use --shard i/n to distribute models across GPUs.
.venv/bin/python -m scripts.score --gpu 0 --shard 0/1 --tiers 1,2

# Score the generative control task.
.venv/bin/python -m scripts.score --gpu 0 --shard 0/1 --tiers 1,2 \
  --mode generative --benchmarks mgsm

# Fit the response model and regenerate the analysis records.
.venv/bin/python -m scripts.analyse

```

Scoring is resumable: each model–benchmark–language shard is written independently, and completed shards are skipped on later runs. The analysis refuses incomplete response tensors by default because missing cells would break the fully crossed design.

## Repository structure

```text
src/mirage/                 Identification, IRT, simulation, and scoring code
tests/                      Numerical checks for the theoretical results
scripts/                    Data, scoring, analysis, and artifact entry points
configs/                    Benchmark and model specifications
results/runs/analysis/      Committed empirical records
results/runs/ablation/      Robustness and alternative-model records
results/runs/toy/           Synthetic claim ledger and generated figures
```

## Reproducibility safeguards

- Benchmark coordinates, revisions, language sets, item identifiers, and field mappings are declared in [`configs/benchmarks.yaml`](configs/benchmarks.yaml).
- The crossed design is enforced by intersecting item identifiers across languages, emitting a canonical order, and realigning by identifier.
- Multiple-choice answers use constrained answer-token decoding, preventing language-correlated parsing failures from entering the response tensor as apparent difficulty.
- Statistical tests control the false discovery rate across the complete item–language grid.
- Tables and figures are generated from committed JSON records.
- A dedicated consistency check detects stale or unsupported numerical claims in the manuscript source.

## Scope and limitations

- The formal results apply to the stated Rasch and 2PL model classes; guessing, multidimensional ability, and option-order effects are handled empirically, not covered by the theorems.
- The panel contains 19 open-weight models no larger than 9.2B parameters. Results need not transfer unchanged to larger or proprietary systems.
- Item-level drift is noisy with only 19 model responses per cell, so the main empirical conclusion is framed as a breakeven sensitivity analysis rather than as a point estimate of uniform drift.
- The headline percentage is calibrated to constrained answer-letter scoring. Alternative scoring preserves the broad language ordering but changes the measured drift dispersion.
- Public benchmark revisions can evolve; the declared configuration and committed result records define the reviewed artifact.

## Ethics and intended use

The study uses public benchmarks and public model checkpoints. It introduces no human-subject data, new annotation, or personal data. Its purpose is not to deny cross-lingual disparities, but to separate claims supported by the measurement design from claims that require additional assumptions. The recommended use is to report a breakeven drift value alongside every cross-lingual gap.
