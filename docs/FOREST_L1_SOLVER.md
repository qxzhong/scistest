# Forest leaf L1 optimization

For fixed tree partitions, the hunting objective is
`||A theta - 0.5||^2 + eta ||theta||_1 - n/4`.
The primary solver uses Lasso with `alpha=eta/(2n)`, no intercept, and a
maximum of 100,000 coordinate-descent iterations in the default RF profile.

Different trees produce structurally dependent columns: summing all leaf
columns within any tree produces the same subject-level integrated score.
This redundancy, additional correlated partitions, and disparate column
scales can make cyclic coordinate descent converge slowly. A convergence
warning is not evidence that the conditional-independence test failed; it
means that the numerical fit has not yet met its optimality criterion.

The fallback preserves the objective, penalty, forest partitions and seeds:

1. Finish the primary fit and retain its coefficients as a warm start.
2. Independently compute a feasible-dual lower bound and the primal-dual gap.
3. On a warning or excessive gap, normalize columns and transform the L1
   weights inversely, preserving the original penalty exactly.
4. Prefer Clarabel's interior-point quadratic-programming solver, with
   explicit residual/slack variables to avoid squaring the condition number.
   Install this backend with `pip install -e '.[forest]'`. ADMM remains an
   alternative when that optional backend is unavailable. Neither backend
   adds a statistical ridge penalty.
5. Accept a solution only when the independently evaluated original Lasso
   gap is at most `solver_tolerance * dot(y,y)/n`, the original threshold.

Both optimizers and their convergence certificates are recorded. Failures
remain explicit if neither solver reaches the required gap; warnings are
never silently counted as successful fits.

The RF simulation, application and cluster recovery programs are excluded
from this upload tree. Only the estimator and its numerical solver are
distributed here. Historical experiment scripts and outputs remain outside
the package.

References: scikit-learn Lasso documentation; Boyd et al. (2011),
*Distributed Optimization and Statistical Learning via the Alternating
Direction Method of Multipliers*.
