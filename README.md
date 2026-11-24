# scistest: conditional independence testing with survival data

`scistest` tests **U ⟂ X | Z** from right-censored observations
`(X, Z, T, event)`, where `T = min(U, C)` and `event = 1[U <= C]`.
This research release follows **main6.tex** of *Conditional Independence
Testing with Survival Data* by Qixian Zhong and Rajen Shah.

Version **0.2.0** refits the null hazard on the evaluation half before scoring.
The hunting hazard is used only to learn the direction. The conditional
engression model is also fitted on the evaluation half. Preprocessing constants
are estimated on the hunting half; score moments use the population variance
convention (`ddof=0`), and inference uses a one-sided normal tail.

## Install

Python 3.10 or newer is required. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# Reproduction scripts, Framingham loader and plotting:
python -m pip install -e '.[dev,experiments]'
```

Engression is pinned to 0.1.15, the version used by the cluster implementation.
The distribution and primary import are both named `scistest`. The old
`scitest` namespace and `sci_test` spelling remain available for compatibility.

## Test your data

```python
from scistest import scis_test

result = scis_test(X, Z, T, event, random_state=1)
print(result.test_statistic, result.p_value)
print(result.as_dict())
```

`X` and `Z` are numeric vectors or matrices with one row per subject; `T` and
`event` are vectors. Inputs must be finite, times nonnegative, and event values
0/1. `scisTest` is an alias for `scis_test`. Zero or nonfinite score variance
raises an error instead of silently producing a p-value.

Configure networks with `HazardConfig`, `DirectionConfig`, `GeneratorConfig`
and `SciTestConfig`. The default direction objective is
`sum(V**2) - sum(V) + l1_penalty * ||weights||_1`.
`DirectionConfig(objective="standardized")` selects the standardized hunting
objective and best-epoch checkpoint used in the Framingham application.

## Optional RF estimators

Install the certified forest leaf solver and select RF for both hazards
and the hunting direction:

```bash
python -m pip install -e '.[forest]'
```

```python
from scistest import scis_test

result = scis_test(X, Z, T, event, estimator="rf", random_state=1)
```

The default `estimator="nn"` retains neural hazards and direction. Both
options implement **two data subsets**: the pilot hazard and direction use
the hunting half; the scoring hazard `lambda_E` and conditional engression
generator use the evaluation half. RF does not replace engression.

RF first constructs tree partitions and then refits their leaf contributions
under main6's penalized losses: mean survival negative log likelihood for
each hazard, and `sum(V**2 - V) + eta_phi * ||gamma||_1` for the direction.
The forest prediction sums effective leaf contributions; L1 penalizes these
contributions, rather than conventional averaged leaf values.

Omitted RF configurations use the total input size `n = len(T)`: 32 trees
for both hazards, 16 trees for the direction, maximum depth 3, minimum leaf
size 20, feature fraction 0.8, and penalties `5/n` and `50/n`, respectively.
The two hazards use the same configuration and are fitted separately.
The generator retains the NN simulation formulas below. Numerical integration
and conditional sampling both use 100 points/draws by default.

Use `forest_configs(n)` to obtain and customize this profile. Explicit
`RandomForestHazardConfig` / `RandomForestDirectionConfig` objects override
defaults independently, including when used to select mixed estimator types;
their static dataclass defaults do not resolve a sample size automatically.
The old `forest_simulation_configs` helper remains a compatibility alias for
the profile constructor; it runs no simulation.

RF simulation and data-application runners are deliberately excluded from
this upload tree. The existing reproduction programs below use NN only.
See [the two-split audit](docs/TWO_SPLIT_AUDIT.md) for equation mappings and
numerical implementation details, and [the solver notes](docs/FOREST_L1_SOLVER.md)
for convergence checks.

When configuration objects are omitted, `scis_test` automatically uses the
selected estimator's profile at `n = len(T)` **before splitting**. Each omitted
configuration is resolved independently: for example, supplying only
`direction_config` leaves hazard and generator settings dependent on `n`.
An explicitly supplied configuration object is used as supplied, including
its dataclass defaults. The profile helpers are also available explicitly:

```python
from scistest.manuscript import simulation_configs, framingham_configs

result = scis_test(X, Z, T, event, random_state=1,
                   **simulation_configs(len(T)))
# For the real-data application instead use **framingham_configs().
```

The default simulation profile uses the total input size `n`: depth `floor(log10(n))`,
width `floor(2*sqrt(n))`, learning rate `2/n`, L1 penalty `5/n` for hazard and
direction, `floor(50*log(n)/log(20))` epochs, and engression layer parameter
`1 + floor(log10(n))`. Integration uses **100 points by default**, following
the manuscript and the author's explicit choice. Historical cluster figure
runs used a different integration grid; see [reproduction notes](docs/MAIN6.md).

## Split aggregation

```python
from scistest import aggregated_scis_test

aggregate = aggregated_scis_test(X, Z, T, event,
                                  K=3, J=10, L=3, random_state=1)
print(aggregate.p_value_decimal)
# Pass estimator="rf" to use RF for every underlying two-split test.
```

Each of `J` permutations supplies `K` disjoint blocks of size `floor(n/K)`.
For omitted configurations, full-data fits use `n` and calibration-subsample
fits use their own size `floor(n/K)` in the formulas. Explicit configurations
are shared unchanged across all fits.
The method refits the complete test on each block and split, pools ranks of
all calibration statistics, and computes a Gaussian-kernel-smoothed tail.
The unsmoothed empirical tail and log probabilities are retained. Generic
`aggregate_p_value` and `calibrate_aggregated_p_value` interfaces also accept
other statistics or already computed matrices. The Framingham reproduction
script reuses one calibration distribution per covariate across 1,000
full-data repetitions, as in main6.

## Repository layout

```text
src/scistest/              installable method, aggregation and paper profiles
src/scitest/               compatibility with the old local package name
simulations/               four simulation groups, comparators and plotting
  protocol.py              DGPs, hypothesis definitions and deterministic seeds
  run.py                   checkpointed execution (local or Slurm task)
  comparators/             deepTest and R vim/GCM/PCM adapters
  summarize.py, plot.py     summaries and figures from newly computed fits
dataApplication/           Framingham data loading, repeated aggregation and plot
examples/                  quick demo and notebook with cleared outputs
scripts/run_simulation.py  small illustrative experiment, not the paper protocol
tests/                     aggregation, method and manuscript regressions
docs/                      reproduction details and source fingerprints
```

## Reproduce main6

Run these commands **from the repository root** after installation:

```bash
# Inspect the full 24,000-task simulation plan without fitting:
python -m simulations.run --dry-run
# Short execution check (reduced settings, unsuitable for reported results):
python -m simulations.run --smoke --output /tmp/scistest-smoke
# Paper simulations; can be split by --group and --sample-sizes:
python -m simulations.run --workers 4 --output results/simulations
python -m simulations.summarize --output results/simulations
python -m simulations.plot --output results/simulations

# Framingham: 18,540 fits, J=10, m=1552, L=3, R=1000:
python -m dataApplication.run_framingham --dry-run
python -m dataApplication.run_framingham --workers 4
python -m dataApplication.plot
```

For comparison methods and the complete-case data checks, see
[simulations/README.md](simulations/README.md) and
[dataApplication/README.md](dataApplication/README.md).
These scripts compute results when invoked; **no fitted models, experiment
results, source datasets or generated figures are distributed here**.

## Develop and build

```bash
python -m pip install -e '.[dev,experiments,forest]'
pytest
ruff check .
python -m build
python -m twine check dist/*
```

The wheel contains the importable library; the source distribution additionally
contains the reproduction scripts, tests and documentation. Neither command
publishes to GitHub or PyPI. Keep generated `dist/` files out of Git.

MIT License. Aggregation follows Guo and Shah (2025), *Rank-transformed
subsampling: inference for multiple data splitting and exchangeable p-values*,
JRSS Series B 87(1), 256–286.
