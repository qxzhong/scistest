# Simulations

From the repository root, install `pip install -e '.[experiments]'`.
`python -m simulations.run` defaults to 200 repetitions of 120 cells (24,000
fits), n = 200,400,600,800,1000, one worker, and **100 integration points**.
`--dry-run` performs no fits or writes. `--smoke` runs one cell and one
replication with one training epoch and 8 integration points/generator draws.
Smoke records are labeled and refused by the manuscript plotting program.

| Group | Study setting | Settings |
| --- | --- | --- |
| case1 | survival-method comparison | six hypotheses, nonlinear correlated W |
| case3a | complete-case comparison (a) | six a values, marginal event null |
| case3b | complete-case comparison (b) | six a values, conditional event null |
| case2 | nondegeneracy | six a values, including degenerate a=0 |

The cluster's case numbers are retained to preserve its seed mapping.
`protocol.py` carries the original DGPs and censoring intercept constants.
Case1 uses master seed 2026111703; other groups use 2026111602. Seeds depend
on setting, sample size and repetition, independently of task scheduling.

```bash
python -m simulations.run --group case1 --sample-sizes 200 --workers 2
python -m simulations.run --group case3a --method GCM
python -m simulations.run --group case3b --method PCM
python -m simulations.run --group case1 --method vim
python -m simulations.run --group case1 --method deepTest
python -m simulations.summarize
python -m simulations.plot
```

The last two commands summarize/plot only locally computed checkpoints.
Per-task settings are validated on resume; changed settings require a new
output directory. Failures stop the command and never count as successful
replications. `n_success` in the summary must be 200 for manuscript curves.
At the degenerate boundary, zero score variance can legitimately cause an
error; no replacement p-values are assigned.

For deepTest, install `pip install -e '.[comparators]'`.
`comparators/deep_test.py` retains the cluster implementation; settings are
five folds, width 32, 64 epochs, rho=0.1. For vim/GCM/PCM install R and run
`Rscript simulations/comparators/install.R`; ensure `R_LIBS_USER` points to
that library on subsequent runs. The R installer pins the cluster package
versions. GCM and PCM use **only event=1 observations**, with observed time
as response and random-forest nuisance regressions. vim uses all censored
observations and the AUC settings from the cluster script.

These adapters can run fresh comparisons on the common DGPs. The historical
simulation figures reused comparator curves from an earlier cluster run; exact
old comparator p-values are not bundled or implied by a new run.
`--integration-grid cluster` selects floor(10 sqrt(n)) for the proposed
method to recover the historical simulation grid. Use a separate output
directory for this option; the default remains the manuscript's fixed 100.
