# Conditional Independence Testing with Survival Data

`scistest` implements the conditional-independence test developed for
right-censored outcomes. Given observations

$$
(X_i,Z_i,T_i,\Delta_i),\qquad
T_i=\min(U_i,C_i),\quad \Delta_i=\mathbf 1(U_i\le C_i),
$$

the package tests

$$
H_0: U\perp X\mid Z.
$$

The repository contains the single-split `scis_test` implementation, the
Guo--Shah (2025) rank-transformed subsampling aggregation, a small demo, a
simulation runner, and unit tests. The code is currently a research release
(`0.1.0`); freeze the version and package settings used for any reported
numerical experiment.

The package is maintained by Qixian Zhong and Rajen Shah and distributed
under the MIT License.

## Repository layout

```text
.
├── pyproject.toml
├── src/scistest/
│   ├── core.py          # single-split scisTest and its p-value
│   ├── aggregation.py   # Guo--Shah aggregated p-value
│   ├── config.py        # typed hyperparameter configurations
│   └── simulation.py    # reproducible example data generator
├── examples/
│   ├── demo.py
│   └── demo.ipynb
├── scripts/run_simulation.py
└── tests/
```

## Installation

From this directory, create an isolated environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development tools, use `python -m pip install -e '.[dev]'`.

## Single-split test

```python
from scistest import scis_test

result = scis_test(
    x=X,             # shape (n,) or (n, d_x)
    z=Z,             # shape (n,) or (n, d_z)
    time=T,          # observed min(event time, censoring time), shape (n,)
    event=Delta,     # 1=event observed, 0=right censored, shape (n,)
    random_state=1,
)

print(result.test_statistic)
print(result.p_value)
```

`scisTest` is provided as an alias for `scis_test`. The returned
`SciTestResult` also includes `log_p_value`, the held-out score moments, sample
indices, and the individual orthogonalized scores. Use
`result.as_dict(include_arrays=False)` for JSON-friendly scalar output.

For each split, the package:

1. fits the null hazard \(\lambda(t\mid Z)=\exp\{g(t,Z)\}\) on the hunting sample
   using the penalized negative counting-process log-likelihood;
2. learns \(\phi\), by default with a DNN, by minimizing exactly
   \(\sum_i V_i(\phi)^2-\sum_i V_i(\phi)\);
3. fits engression for the joint conditional distribution
   \(P_{(X,T)\mid Z}\) on the test sample;
4. constructs the orthogonalized held-out scores and reports the one-sided
   standard-normal tail probability.

Preprocessing constants are estimated only on the hunting sample: `X` is
min--max scaled, `Z` is standardized, and observed time is divided by the
maximum hunting-sample time.

Hyperparameters are explicit immutable dataclasses:

```python
from scistest import (
    DirectionConfig,
    GeneratorConfig,
    HazardConfig,
    SciTestConfig,
    scis_test,
)

result = scis_test(
    X, Z, T, Delta,
    random_state=10,
    hazard_config=HazardConfig(hidden_dims=(32,), epochs=100),
    direction_config=DirectionConfig(hidden_dims=(32, 32), epochs=100),
    generator_config=GeneratorConfig(num_layers=2, hidden_dim=100, epochs=100),
    test_config=SciTestConfig(integration_points=100, generator_draws=100),
)
```

### Random-forest direction estimator

The DNN remains the default. To estimate \(\phi\) with a random forest instead,
pass `RandomForestDirectionConfig` as `direction_config`:

```python
from scistest import RandomForestDirectionConfig, scis_test

result = scis_test(
    X, Z, T, Delta,
    random_state=10,
    direction_config=RandomForestDirectionConfig(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=5,
        n_jobs=-1,
    ),
)
```

The forest partitions the `(time, Z, X)` feature space. Conditional on those
partitions, its terminal-node values are obtained by sparse least squares using

\[
\sum_i\{V_i(\phi)^2-V_i(\phi)\}
=\sum_i\{V_i(\phi)-1/2\}^2-n_1/4.
\]

Thus both direction estimators target the same criterion; only the function
class and optimization method differ. The hazard and conditional-generator
estimators are unchanged.

## Aggregated p-value with inputs `(K, J, L)`

The convenience interface refits the full scisTest procedure for all full-data
splits and calibration subsamples:

```python
from scistest import aggregated_scis_test

aggregated = aggregated_scis_test(
    X, Z, T, Delta,
    K=3,     # number of disjoint subsamples per permutation
    J=5,     # independent permutations
    L=6,     # random splits per full sample or subsample
    random_state=20260918,
)

print(aggregated.aggregated_p_value) # smoothed right-tail p-value
print(aggregated.empirical_p_value) # unsmoothed calibration fraction
```

The defaults are `K=3`, `J=5`, and `L=6`, so these three arguments may be
omitted.

Each permutation divides the data into `K` disjoint blocks of common size

\[
\texttt{subsample\_size}=\lfloor n/K\rfloor.
\]

The remaining `n % K` observations are unused in that permutation. Thus the
number of calibration rows is

\[
B=JK.
\]

All \(BL\) subsample statistics are pooled for the randomized rank transform

\[
\widetilde H_{b\ell}=\Phi^{-1}\left\{\frac{R_{b\ell}-1/2}{BL}\right\}.
\]

The full-sample and row-wise calibration aggregates are arithmetic means over
the \(L\) split statistics. The primary result is the Gaussian-kernel-smoothed
right tail used by the authors' MultiSplit implementation. The result also
retains the strict empirical tail, whose resolution is `1 / B`. Because a
publication-scale run fits `(B + 1) * L` complete tests, it can be expensive.

For another asymptotically standard-normal split statistic, use the generic
callback API:

```python
from scistest import aggregate_p_value

def statistic(indices, split_seed):
    # Subset the data, refit the complete base method using split_seed,
    # and return one one-sided standard-normal statistic.
    return my_statistic(indices, split_seed)

aggregated = aggregate_p_value(
    n=len(T),
    statistic=statistic,
    K=3,
    J=5,
    L=6,
    random_state=20260918,
)
```

If the full and subsample statistics have already been computed, call
`calibrate_aggregated_p_value(observed, subsampled, K, J, L, n=n)`. This is
useful for checkpointed cluster runs.

## Demo and simulation

Run a quick single-split example:

```bash
python examples/demo.py
```

For an interactive walkthrough of the DNN direction, random-forest direction,
and `K`-based aggregation interfaces, open `examples/demo.ipynb`.

Add `--aggregate` to demonstrate the complete aggregation workflow with small
settings. Run a reproducible null/alternative experiment with:

```bash
python scripts/run_simulation.py \
  --repetitions 20 \
  --sample-size 400 \
  --scenario both \
  --output results/simulation.csv
```

Aggregation can be enabled with `--aggregate`; the command-line defaults are
also `--K 3 --J 5 --L 6`. It is much more computationally demanding because
every calibration statistic refits all nuisance models.

## Tests

```bash
pytest
ruff check .
```

## Reference

The aggregation follows Guo, F. Richard and Rajen D. Shah (2025),
“Rank-transformed subsampling: inference for multiple data splitting and
exchangeable p-values,” *Journal of the Royal Statistical Society Series B*,
87(1), 256–286.
