# Changelog

## 0.2.0

- Add `estimator="rf"` to the two-split test and aggregation interfaces;
  use main6's RF table defaults (32/16 trees, leaf L1 penalties 5/n and 50/n).
- Keep RF experiment orchestration outside the upload tree; retain NN runners.
- Preserve inference under small positive score rescalings by testing for
  zero variance without an absolute standard-deviation cutoff.

- Certify L1 leaf fits independently and use a same-objective QP/ADMM fallback
  when coordinate descent does not converge.

- Use the simulation sample-size formulas automatically for omitted method
  configurations; calibration subsamples use their own input sample size.

- Refit the scoring hazard on the evaluation half, following main6.
- Restore the main6 direction L1 penalty; support quadratic simulation and
  standardized application objectives.
- Add sample-size simulation profiles and fixed Framingham settings.
- Add result-free simulation and data-application reproduction programs.
- Use the scistest distribution/import name and preserve scitest compatibility.
- Retain the current GitHub random-forest estimator and aggregation interfaces.
