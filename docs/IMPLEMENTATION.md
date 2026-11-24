# Implementation and provenance

Updated on 2026-11-23. `source_provenance.json` records source hashes and the
cluster directories used. The original source files were backed up outside
the repository before editing.

The starting library was reconciled with the current public qxzhong/scistest
source archive, retaining its scistest naming, K/J/L aggregation interface,
MIT license and optional random-forest direction estimator. The old local
scitest namespace still delegates to the updated method; its Q-based
aggregation module is retained for existing local callers.

## Implementation overview

| Method operation | Source in this repository |
| --- | --- |
| Training-half pilot hazard and hunting direction | src/scistest/core.py |
| Evaluation-half generator and refitted lambda_E | src/scistest/core.py |
| Risk-set Monte Carlo centering and orthogonal score | src/scistest/core.py |
| Formula hyperparameters | src/scistest/manuscript.py: simulation_configs |
| Fixed real-data settings | src/scistest/manuscript.py: framingham_configs |
| Optional RF hazards and direction, with size-dependent defaults | src/scistest/manuscript.py: forest_configs; src/scistest/_forest.py |
| Four main-text DGP groups | simulations/protocol.py |
| vim, GCM, PCM, deepTest | simulations/comparators/ |
| Complete-case Framingham and repeated calibration | dataApplication/run_framingham.py |

The three-split procedure in the paper is used for theoretical exposition; the
numerical experiments and this library's primary interface implement the
paper's practical **two-split** procedure. The commented older supplementary
simulation/application text in earlier drafts is not treated as a second numerical
protocol.

Omitted NN configurations resolve to `simulation_configs(n)` and RF
configurations to `forest_configs(n)` using
the total input size before splitting. Each calibration subsample resolves
its own size-dependent defaults. Explicit configuration objects are retained;
the Framingham runner continues to pass its fixed application profile.
The RF estimator is selected with `estimator="rf"` in `scis_test` or
`aggregated_scis_test`. RF simulation/data-application orchestration is not
included. The NN reproduction programs remain available.

The detailed equation audit and numerical conventions are in
[TWO_SPLIT_AUDIT.md](TWO_SPLIT_AUDIT.md).

## Explicit decisions and reproducibility limits

1. **Integration:** the paper specifies 100 points, while the historical
   cluster figure run used floor(10 sqrt(n)). The author explicitly selected
   fixed 100 points for this repository. No figure or p-value from the
   historical run is claimed to be reproduced by the default grid.
2. **Objectives:** simulations use the penalized quadratic hunting objective;
   Framingham uses the penalized standardized objective and best-epoch
   selection, matching the paper and cluster implementation.
3. **Hazard clipping:** both event and integrated-hazard terms now use the
   same clipped log hazard, following the corrected simulation fitter.
   The older Framingham cluster fitter left the event term unclipped. This
   correction can change fits when clipping is active.
4. **Conditional draws:** the generator uses independent one-sample
   predictions at repeated Z values, as in the cluster implementation.
   Engression is pinned to 0.1.15. Exact numerical agreement also depends on
   the remaining library versions, platform and floating-point behavior.
5. **Comparators:** historical cluster simulation figures reused comparator
   results from earlier runs. The repository includes executable adapters,
   not those historical outputs. New comparisons use the shared DGPs.
6. **Result exclusion:** no source dataset, final p-value table, plot, fitted
   checkpoint or cached notebook output is distributed. Scripts generate
   those artifacts only when invoked. Build artifacts are excluded from Git.

## Release preparation

Version 0.2.0 distinguishes the evaluation-hazard method from the previous
0.1.0 research release. Run pytest, Ruff, source/wheel builds and Twine checks
before a release. The wheel contains the library and compatibility namespace;
the source archive also includes experiment scripts and documentation.
The upstream release-triggered PyPI workflow is retained. It uses GitHub's
configured pypi environment and trusted publishing when a release is published;
this preparation does not trigger it or upload anything to GitHub/PyPI.
