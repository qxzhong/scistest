"""Forest partitions, followed by convex leaf fitting under manuscript losses.

The optimized coefficients are effective leaf contributions theta: the forest
prediction is the SUM of these contributions. Equivalently, conventional
averaged forest leaf values are n_trees * theta. L1 penalizes sum(abs(theta)).
Tree partitions are auxiliary, data-adaptive candidates, not global minimizers
over tree structures. Quadrature observations from a subject stay in the same
sample split. No outcome from the evaluation half enters the hunting fit.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix, hstack
from scipy.sparse.linalg import lsqr
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from ._lasso import lasso_gap, solve_lasso_certified
from ._models import RandomForestDirection


def leaf_design(forest, features):
    blocks, nodes_per_tree = [], []
    for tree in forest.estimators_:
        nodes = tree.apply(features)
        unique, columns = np.unique(nodes, return_inverse=True)
        blocks.append(
            csr_matrix(
                (np.ones(len(features)), (np.arange(len(features)), columns)),
                shape=(len(features), len(unique)),
            )
        )
        nodes_per_tree.append(unique)
    return hstack(blocks, format="csr"), nodes_per_tree


def leaf_arrays(forest, nodes_per_tree, theta):
    values, start = [], 0
    for tree, nodes in zip(forest.estimators_, nodes_per_tree, strict=True):
        array = np.zeros(tree.tree_.node_count)
        array[nodes] = theta[start : start + len(nodes)] * len(forest.estimators_)
        values.append(array)
        start += len(nodes)
    return values


def features_grid(z, t, points, x=None):
    n = len(t)
    grid = t[:, None] * torch.arange(1, points + 1, dtype=t.dtype)[None, :] / points
    parts = [
        grid.reshape(-1, 1),
        z[:, None, :].expand(n, points, z.shape[1]).reshape(-1, z.shape[1]),
    ]
    if x is not None:
        parts.append(x[:, None, :].expand(n, points, x.shape[1]).reshape(-1, x.shape[1]))
    return torch.cat(parts, dim=1).numpy(), grid


def make_forest(config, seed, *, criterion="squared_error"):
    return RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        bootstrap=config.bootstrap,
        n_jobs=config.n_jobs,
        random_state=seed,
        criterion=criterion,
    )


def likelihood_loss_gradient(theta, design, exposure, event_design, penalty):
    log_hazard = np.asarray(design @ theta)
    integrated = exposure * np.exp(log_hazard)
    value = integrated.sum() - event_design @ theta + penalty * np.abs(theta).sum()
    gradient = np.asarray(design.T @ integrated).ravel() - event_design
    return float(value), gradient


class RandomForestHazard:
    """Positive hazard exp(forest log hazard), with separately optimized leaves."""

    def __init__(self, forest, leaf_values, diagnostics):
        self.forest, self.leaf_values, self.diagnostics = forest, leaf_values, diagnostics

    def log_hazard(self, time, z):
        features = torch.cat((time.reshape(-1, 1), z), dim=1).numpy()
        result = np.zeros(len(features))
        for tree, values in zip(self.forest.estimators_, self.leaf_values, strict=True):
            result += values[tree.apply(features)]
        result /= len(self.forest.estimators_)
        return torch.as_tensor(result, dtype=time.dtype)

    def hazard(self, time, z):
        return torch.exp(self.log_hazard(time, z))


def fit_hazard(z, time, event, config, points, seed):
    features, _ = features_grid(z, time, points)
    n = len(time)
    exposure = np.repeat(time.numpy().astype(float) / points, points)
    events = np.zeros((n, points))
    events[:, -1] = event.numpy()
    events = events.ravel()
    if events.sum() == 0:
        raise ValueError("Forest hazard needs an observed event in its fitting sample")
    positive = exposure > 0
    target = events[positive] / exposure[positive]
    forest = make_forest(config, seed, criterion="poisson")
    forest.fit(features[positive], target, sample_weight=exposure[positive])
    design, nodes = leaf_design(forest, features)
    event_design = np.asarray(design.T @ events).ravel() / n
    exposure = exposure / n
    dimension = design.shape[1]
    # Each tree contributes at most bound/n_trees to g; g is therefore bounded.
    bound = config.max_abs_log_hazard / config.n_estimators

    def objective(uv):
        theta = uv[:dimension] - uv[dimension:]
        value, gradient = likelihood_loss_gradient(theta, design, exposure, event_design, 0)
        value += config.l1_penalty * uv.sum()
        return value, np.r_[gradient + config.l1_penalty, -gradient + config.l1_penalty]

    initial = np.zeros(2 * dimension)
    solution = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0, bound)] * (2 * dimension),
        options={
            "maxiter": config.solver_max_iterations,
            "ftol": 1e-12,
            "gtol": config.solver_tolerance,
            "maxls": 40,
            "maxcor": 10,
        },
    )
    theta = solution.x[:dimension] - solution.x[dimension:]
    # L-BFGS-B's objective-change stop need not satisfy the L1 KKT conditions
    # with redundant forest leaves. Polish by exact bounded coordinate updates.
    columns = design.tocsc()
    integrated = exposure * np.exp(design @ theta)
    sweeps = 0
    residual = np.full(dimension, np.inf)
    for _ in range(config.solver_max_iterations):
        sweeps += 1
        for column in range(dimension):
            rows = columns.indices[columns.indptr[column] : columns.indptr[column + 1]]
            current = theta[column]
            baseline = integrated[rows].sum() * np.exp(-current)
            count = event_design[column]
            if baseline <= 0:
                updated = 0.0
            elif count - baseline > config.l1_penalty:
                updated = np.log((count - config.l1_penalty) / baseline)
            elif baseline - count > config.l1_penalty:
                updated = np.log((count + config.l1_penalty) / baseline)
            else:
                updated = 0.0
            updated = float(np.clip(updated, -bound, bound))
            theta[column] = updated
            integrated[rows] *= np.exp(updated - current)
        gradient = np.asarray(design.T @ integrated).ravel() - event_design
        residual = gradient + config.l1_penalty * np.sign(theta)
        zero = theta == 0
        residual[zero] = np.sign(gradient[zero]) * np.maximum(
            np.abs(gradient[zero]) - config.l1_penalty, 0
        )
        residual[(theta <= -bound + 1e-8) & (residual > 0)] = 0
        residual[(theta >= bound - 1e-8) & (residual < 0)] = 0
        if np.abs(residual).max() <= config.solver_tolerance:
            break
    if np.abs(residual).max() > config.solver_tolerance:
        raise RuntimeError(
            f"Forest hazard KKT residual {np.abs(residual).max():.3g} "
            f"exceeds tolerance {config.solver_tolerance}"
        )
    value, _ = likelihood_loss_gradient(theta, design, exposure, event_design, config.l1_penalty)
    diagnostics = {
        "objective": value,
        "initial_objective": float(exposure.sum()),
        "iterations": solution.nit,
        "coordinate_polish_sweeps": sweeps,
        "l1_norm": float(np.abs(theta).sum()),
        "active_leaves": int((np.abs(theta) > 1e-8).sum()),
        "total_leaves": len(theta),
        "kkt_residual": float(np.abs(residual).max()),
    }
    if not np.isfinite(value) or value > exposure.sum() + 1e-7:
        raise RuntimeError("Forest hazard optimization failed to improve its initial loss")
    return RandomForestHazard(forest, leaf_arrays(forest, nodes, theta), diagnostics)


def fit_direction(hazard, z, x, time, event, config, points, seed):
    features, grid = features_grid(z, time, points, x)
    n = len(time)
    z_grid = z[:, None, :].expand(n, points, z.shape[1]).reshape(-1, z.shape[1])
    with torch.no_grad():
        coefficients = -hazard.hazard(grid.reshape(-1), z_grid).reshape(n, points)
        coefficients = coefficients * time[:, None] / points
        coefficients[:, -1] += event
    flat = coefficients.numpy().astype(float).ravel()
    forest = make_forest(config, seed)
    forest.fit(features, flat)
    point_design, nodes = leaf_design(forest, features)
    aggregation = coo_matrix(
        (flat, (np.repeat(np.arange(n), points), np.arange(n * points))), shape=(n, n * points)
    ).tocsr()
    design = aggregation @ point_design
    if config.l1_penalty > 0:
        # sklearn uses ||A theta-y||^2/(2n) + alpha ||theta||_1.
        model = Lasso(
            alpha=config.l1_penalty / (2 * n),
            fit_intercept=False,
            max_iter=config.solver_max_iterations or 20000,
            tol=config.solver_tolerance,
            selection="cyclic",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(design, np.full(n, 0.5))
        theta = model.coef_
        iterations = int(model.n_iter_)
        # Recompute the certificate independently; do not accept a warning
        # simply because a finite coefficient vector was returned.
        target = np.full(n, 0.5)
        dual_gap = lasso_gap(design, target, theta, config.l1_penalty / 2)
        gap_tolerance = config.solver_tolerance * float(target @ target) / n
        solver_details = {
            "solver": "coordinate_descent",
            "dual_gap": dual_gap,
            "gap_tolerance": gap_tolerance,
        }
        needs_fallback = (
            any(issubclass(w.category, ConvergenceWarning) for w in caught)
            or not np.isfinite(dual_gap)
            or dual_gap > gap_tolerance
        )
        if needs_fallback:
            theta, solver_details = solve_lasso_certified(
                design,
                target,
                config.l1_penalty / 2,
                config.solver_tolerance,
                initial=theta,
                max_iterations=max(config.solver_max_iterations or 20000, 100000),
            )
            solver_details["primary_iterations"] = iterations
            solver_details["primary_dual_gap"] = dual_gap
            iterations += solver_details["iterations"]
            dual_gap = solver_details["dual_gap"]
    else:
        solution = lsqr(
            design,
            np.full(n, 0.5),
            atol=config.solver_tolerance,
            btol=config.solver_tolerance,
            iter_lim=config.solver_max_iterations,
        )
        theta, iterations, dual_gap = solution[0], solution[2], None
        solver_details = {"solver": "lsqr"}
    scores = np.asarray(design @ theta).ravel()
    value = float((scores**2 - scores).sum() + config.l1_penalty * np.abs(theta).sum())
    if not np.isfinite(value) or value > 1e-7:
        raise RuntimeError("Forest direction optimization failed to improve zero direction")
    direction = RandomForestDirection(forest, leaf_arrays(forest, nodes, theta))
    direction.diagnostics = {
        "objective": value,
        "l1_norm": float(np.abs(theta).sum()),
        "active_leaves": int((np.abs(theta) > 1e-8).sum()),
        "total_leaves": len(theta),
        "iterations": iterations,
        "dual_gap": dual_gap,
        **solver_details,
    }
    return direction, value
