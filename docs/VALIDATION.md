# Validation

## 2026-11-23: optional RF and two-split audit

- 42 tests passed in the isolated Python 3.10 environment, including a real
  RF fit through `estimator="rf"`, independent checks of both penalized leaf
  loss formulas, use of the evaluation hazard, size-dependent RF defaults
  in aggregation, direct risk-set score calculations, and inference under
  small positive direction rescalings.
- Ruff checks passed. The existing NN simulation dry run still enumerates
  24,000 fits and the NN application dry run enumerates 18,540 fits.
- Wheel and source archive builds and Twine checks passed. Both archives
  were inspected for excluded RF experiment runners, data, results and
  caches. A separate wheel installation verified the exported RF selector,
  size-dependent profile and compatibility imports.
- RF solver sources were verified against the cluster; the statistical
  solver files remain identical. Default hazard trees/penalty were corrected
  to the manuscript table, and a public NN/RF selector was added.
- RF experiment runners were moved outside the upload tree; its NN runners
  remain. No empirical experiment output, dataset or figure was added.
- See `TWO_SPLIT_AUDIT.md` for equation mappings and numerical limitations.

## 2026-11-20: preceding validation

- Python 3.10 isolated environment; Engression 0.1.15.
- 32 tests passed, including real hazard refitting and both direction
  objectives, aggregation, source-seed/DGP regression checks, and Framingham
  encoding/subsampling checks.
- Default-configuration regressions verify equivalence to the explicit
  simulation profile, preservation of partial overrides, and independent
  sample-size resolution for full data and calibration subsamples.
- Leaf-Lasso regressions additionally verify the unchanged L1 objective
  after numerical normalization, rank-deficient bases, and certified fallback
  operation. Eight difficult cluster failures passed a separate recovery
  preflight with their original hyperparameters and seeds.
- Ruff checks passed.
- Simulation dry run resolves 24,000 tasks with the default 100-point grid.
- One reduced simulation completed and its checkpoint summary was generated
  outside the repository.
- Framingham dry run resolves 18,540 fits; data initialization verified 4,658
  complete cases and 1,465 events. One full-data, full-epoch single-split fit
  completed with evaluation-hazard scoring and the standardized objective.
- Both R scripts parsed. Optional R comparison methods were not executed:
  this machine does not have survML/comets installed. The optional deepTest
  adapter was not executed as part of these checks.
- Source archive and wheel built successfully; Twine metadata checks passed.
  The wheel was installed into a separate directory and imports checked.
- Archives inspected for result/cache artifacts; notebook outputs are empty.

The complete Monte Carlo study and the 1,000-repeat Framingham analysis were
not rerun. No validation-generated datasets, p-values or figures are stored
in this upload directory. These checks establish execution and packaging,
not agreement of all empirical rejection rates with the historical figures.
