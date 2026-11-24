# main6 two-split implementation audit

Audited on 2026-11-23 against the local `tex/main6.tex`. This audit covers
the practical two-split procedure, its NN implementation, and the optional
RF estimators. The three-split theoretical algorithm is not implemented.
The manuscript was not edited as part of this audit. Exact source hashes
are recorded in `source_provenance.json`.

## RF source and public interface

The local `_forest.py` and `_lasso.py` were verified byte-for-byte against
the cluster implementation used by the RF Case 2/3 run and imported by the
subsequent Framingham RF retuning run. The statistical solvers are retained.

```python
from scistest import scis_test

nn = scis_test(X, Z, T, event, estimator="nn", random_state=1)
rf = scis_test(X, Z, T, event, estimator="rf", random_state=1)
```

The selector supplies defaults for omitted configurations. Each explicit
configuration independently overrides its default, so mixed estimator types
remain possible. `aggregated_scis_test` forwards the selector to every
underlying two-split fit. `forest_configs(n)` returns editable RF settings;
`forest_simulation_configs` is retained only as an alias for this constructor.

## Equation mapping

| Manuscript requirement | Implementation | Audit result |
| --- | --- | --- |
| Two disjoint subsets, default 50:50 | `core._prepare_data` | All subjects assigned once; no third fitting subset |
| Preprocessing learned on hunting subset | `core._prepare_data` | X min/max, Z mean/SD, and maximum observed time use hunting data only |
| Pilot hazard and direction learned on hunting subset | `core.scis_test`, `_fit_hazard`, `_fit_direction` | Evaluation outcomes do not enter either hunting fit |
| Refit the same hazard specification on evaluation subset | `core.scis_test` | The same configuration is passed to a separate fit; its output scores the test |
| Fit the conditional law of (X,T) given Z on evaluation subset | `_models.EngressionGenerator.fit` | True for both NN and RF; RF does not replace the generator |
| Risk-set projection, `eq:debiased_m` | `core._orthogonalized_scores` | Conditional-draw average weighted by `1[s <= generated_T]`; zero denominator maps to zero |
| Orthogonal score, `eq:evaluation-subject-score` | `core._orthogonalized_scores` | Event term minus integral of evaluation hazard times centered direction |
| Riemann sum with 100 points | `core._right_riemann_grid` | Uses right endpoints `j*T_i/100`, j=1,...,100, with weight T_i/100 |
| 100 conditional draws | `manuscript` profiles | Same draws reused over a subject's time grid, including the event endpoint |
| Empirical variance with divisor n2 | `core.scis_test` | `ddof=0`, with no added variance ridge |
| Statistic sqrt(n2) W and one-sided p-value | `core.scis_test` | sqrt(n2) times score mean/SD; `norm.sf` and `norm.logsf` |
| NN mean survival negative log likelihood + L1 | `core._fit_hazard` | Fits separately on the two subsets with the same loss specification |
| NN sum(V_i^2 - V_i) + L1 | `core._fit_dnn_direction` | Default simulation direction objective; sums are not replaced by means |
| Application standardized hunting objective | `framingham_configs`, `core._fit_dnn_direction` | Explicit fixed NN application profile; independent of RF defaults |
| RF mean survival negative log likelihood + L1, `eq:rf-hazard-loss` | `_forest.fit_hazard` | Exposure and event contributions divided by fitting-subset size; penalty not divided again |
| RF sum(V_i^2 - V_i) + L1, `eq:rf-direction-loss` | `_forest.fit_direction` | Lasso response 1/2, no intercept, alpha=eta_phi/(2*n1) |
| RF sum of effective leaf contributions | `_forest.leaf_arrays`, model evaluation | Stored conventional leaf values equal tree count times effective contribution; averaging recovers the specified sum |
| RF regularization of effective leaves | `_forest` | Penalty is on effective coefficients, not their tree-count-scaled storage values |

The RF direction conversion uses

\[
\sum_i\{(A\gamma)_i^2-(A\gamma)_i\}
+\eta_\phi\|\gamma\|_1
=\|A\gamma-\tfrac12\mathbf1\|_2^2
+\eta_\phi\|\gamma\|_1-\frac{n_1}{4}.
\]

Thus sklearn's objective `||A gamma - y||^2/(2*n1) + alpha*||gamma||_1`
requires `alpha=eta_phi/(2*n1)`. The certified fallback solves the same
objective and does not introduce a ridge penalty.

## Hyperparameters

Here n is the total input size before the hunting/evaluation split. In
aggregation, each calibration subsample uses its own input size when settings
are omitted. Explicit configurations are preserved unchanged.

| Omitted settings | NN defaults | RF defaults |
| --- | --- | --- |
| Hazard and direction architecture | floor(log10(n)) hidden layers, each width floor(2 sqrt(n)) | Hazard: 32 trees; direction: 16 trees; both depth 3 |
| Minimum leaf size / feature fraction | Not applicable | 20 / 0.8, bootstrap enabled |
| Hazard L1 penalty | 5/n | 5/n, for both hazards |
| Direction L1 penalty | 5/n | 50/n |
| Generator architecture | Width floor(2 sqrt(n)); layer parameter 1+floor(log10(n)) | Same engression generator |
| NN optimizer | Adam, learning rate 2/n, floor(50 log20(n)) epochs | Same settings for generator only |
| Hazard/generator minibatch size | 128 | 128 for generator; RF leaf losses use all fitting subjects |
| Integration / generator draws | 100 / 100 | 100 / 100 |

The original RF configuration factory defaulted to 16 hazard trees and 2/n
hazard penalty. Both were corrected to match `tab:rf-parameters` (32 and 5/n).
Explicit configuration dataclasses have static defaults; constructing one
counts as an explicit override and does not trigger size-based resolution.

The existing NN application runner passes `framingham_configs()` explicitly:
32-unit hazard layer, two 32-unit direction layers, 100 epochs,
learning rates .01/.01/.005, and hazard/direction L1 .001. Its repeated
aggregation uses J=10, m=1552, B=30, L=3 and 1,000 full-data repetitions.
L=3 means three repeated two-split tests; it does not mean three data subsets.
Generic aggregation keeps its existing configurable K/J/L defaults;
`K=3, J=10, L=3` selects the paper's application design for n=4658.

The RF application-specific table (64 hazard trees, 8 direction trees and
different leaf sizes/penalties) is not substituted for the generic RF
defaults. RF application reproduction is outside this upload's scope.

## Numerical conventions and limits

The principal algorithm and loss scaling agree with the manuscript. Exact
unconstrained optimization or reproduction of historical figures is not
claimed, for the following concrete reasons:

- RF tree partitions are constructed first using auxiliary forest regression
  objectives; only the leaf coefficients are then optimized under the paper's
  losses. This is not a global optimization over tree structures. Leaf-size
  settings apply to time-grid rows in those partition fits, not to counts of
  distinct subjects. All rows of a subject stay in that subject's data subset.
- The cluster RF hazard solver bounds each effective leaf contribution by
  `15 / n_trees` in absolute value. This ensures bounded log hazards but is
  an additional restriction on the coefficient class beyond the displayed
  unconstrained loss. KKT checks certify the bounded problem. The NN hazard
  also clips log hazards to [-15,15] in both likelihood terms.
- Engression produces unconstrained outputs. As in the cluster code,
  generated times are replaced by their absolute values. This folds negative
  draws rather than sampling from a distribution constrained to positive times.
- The standardized NN direction loss uses 1e-8 inside its training SD and
  retains the epoch with the largest unpenalized standardized hunting score.
  L1 is included in training. This checkpoint convention is inherited from
  the cluster application and is distinct from exact minimization of the
  penalized criterion. It does not affect the RF quadratic leaf loss.
- Zero or nonfinite empirical score variance raises an error. A previous
  absolute SD cutoff of 1e-12 was removed because it could reject a valid
  positive rescaling of a direction. The returned statistic has no such cutoff.
- Historical cluster NN figures used floor(10 sqrt(n)) integration points.
  The author explicitly chose the manuscript's fixed 100-point default for
  this package. The historical grid remains an explicit option in the NN
  runner, and exact historical rejection rates are not claimed here.
- The retained NN simulation DGPs and hypotheses match the described groups.
  The Case 1 censoring intercept is stored at the cluster's full precision,
  0.7818669539671581, whereas the manuscript displays 0.78. This literal
  numerical difference is preserved for the existing seed/DGP protocol;
  it is not an exact implementation of the rounded printed constant.
- The standard-deviation and IQR bandwidth formula has a positive fallback
  when calibration values are degenerate, matching the prior implementation.
  This extends the displayed formula to its zero-bandwidth boundary.

Tests verify formulas, sample routing, scaling, solver certificates and
execution. They do not verify the statistical rate/stability assumptions of
the two-split theorems, or repeat the full numerical experiments.

## Files excluded from the upload

The old `simulations/cluster_case1_rf` and `simulations/cluster_rf_remaining`
directories were archived outside `code/github`. The RF branch and its CLI
options were removed from `simulations/run.py`. The `simulations` and
`dataApplication` directories retain their NN reproduction programs and
comparison methods. No RF application runner, dataset, fitted model, final
simulation result or plot is included. Estimator regression tests are retained.
