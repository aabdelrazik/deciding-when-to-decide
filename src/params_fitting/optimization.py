# src/params_fitting/optimize.py

import numpy as np
from scipy.optimize import minimize
from geneticalgorithm import geneticalgorithm as ga
import pygad
from src.config import *


# Define scaling functions
def scale_params(params: np.ndarray, param_ranges: dict) -> np.ndarray:
    """Scale a parameter vector to [0, 1] range, per-parameter, using param_ranges.

    Args:
        params (np.ndarray): Parameter values, ordered per param_ranges.
        param_ranges (dict): Maps parameter name to a (low, high) bound tuple;
            iteration order must match `params`.

    Returns:
        np.ndarray: Scaled params, same shape as `params`.
    """
    scaled = np.zeros_like(params)
    for i, (key, (low, high)) in enumerate(param_ranges.items()):
        scaled[i] = (params[i] - low) / (high - low)
    return scaled


def unscale_params(scaled_params: np.ndarray, param_ranges: dict) -> np.ndarray:
    """Convert a [0, 1]-scaled parameter vector back to its original range.

    Args:
        scaled_params (np.ndarray): Parameter values in [0, 1], ordered per
            param_ranges.
        param_ranges (dict): Maps parameter name to a (low, high) bound tuple;
            iteration order must match `scaled_params`.

    Returns:
        np.ndarray: Unscaled params, same shape as `scaled_params`.
    """
    original = np.zeros_like(scaled_params)
    for i, (key, (low, high)) in enumerate(param_ranges.items()):
        # Clip to the declared bound. The scale and unscale round trip can put a
        # value fractionally outside the range it was searched in, which then
        # shows up as a fitted parameter its own config says is impossible.
        original[i] = np.clip(scaled_params[i] * (high - low) + low, low, high)
    return original


def scaled_cost_function(scaled_params: np.ndarray, cost_function, param_ranges: dict):
    """Unscale `scaled_params` back to the original range, then evaluate cost_function.

    Args:
        scaled_params (np.ndarray): Parameter values in [0, 1], ordered per
            param_ranges.
        cost_function (Callable[[np.ndarray], float]): Cost function expecting
            unscaled params.
        param_ranges (dict): Maps parameter name to a (low, high) bound tuple.

    Returns:
        float: cost_function's value at the unscaled params.
    """
    original_params = unscale_params(scaled_params, param_ranges)
    return cost_function(original_params)


def refine_with_minimizer(initial_params: np.ndarray, param_ranges: dict, cost_function):
    """Locally refine a candidate solution with scipy's L-BFGS-B, in scaled [0, 1] space,
    then transform the resulting Hessian back to the original parameter units.

    Args:
        initial_params (np.ndarray): Starting parameter values (unscaled), ordered
            per param_ranges.
        param_ranges (dict): Maps parameter name to a (low, high) bound tuple.
        cost_function (Callable[[np.ndarray], float]): Function to minimize,
            expecting unscaled params.

    Returns:
        tuple: (best_params_dict, best_fitness, hessian_original), where
            hessian_original is the inverse-Hessian-derived Hessian estimate
            transformed from scaled to original parameter units. start_hazard,
            if present, is floored to an int.
    """
    initial_params_scaled = scale_params(initial_params, param_ranges)
    bounds_scaled = [(0, 1)] * len(initial_params)

    result = minimize(
        lambda x: scaled_cost_function(x, cost_function, param_ranges),
        x0=initial_params_scaled,
        method="L-BFGS-B",
        bounds=bounds_scaled,
        options={"maxiter": 2000},
        tol=1e-15,
    )
    best_params_scaled = result.x
    best_fitness = result.fun
    best_params = unscale_params(best_params_scaled, param_ranges)
    inv_hess_scaled = result.hess_inv.todense()
    hessian_scaled = np.linalg.inv(inv_hess_scaled)

    # Check eigenvalues in scaled space
    scaling_factors = np.array(
        [param_ranges[key][1] - param_ranges[key][0] for key in param_ranges]
    )
    J = np.diag(1 / scaling_factors)  # Jacobian of scaling transform

    # Transform Hessian to original space
    best_params_dict = {k: v for k, v in zip(param_ranges.keys(), best_params)}
    if "start_hazard" in best_params_dict:
        best_params_dict["start_hazard"] = int(
            np.floor(best_params_dict["start_hazard"])
        )
    hessian_original = J.T @ hessian_scaled @ J

    return best_params_dict, best_fitness, hessian_original


def ga_fit(param_ranges: dict, cost_function, is_minimzer: bool = False) -> tuple:
    """Fit parameters by minimizing cost_function with a genetic algorithm
    (geneticalgorithm.geneticalgorithm), optionally refined with L-BFGS-B.

    Args:
        param_ranges (dict): Maps parameter name to a (low, high) bound tuple.
            Order determines the parameter vector order passed to cost_function.
        cost_function (Callable[[np.ndarray], float]): Function to minimize.
        is_minimzer (bool, optional): If True, refine the GA's best solution
            with refine_with_minimizer and return its (params, fitness, Hessian)
            instead. Defaults to False.

    Returns:
        tuple: (best_params_dict, best_fitness, hessian_placeholder). When
            is_minimzer is False, hessian_placeholder is a zero matrix (GA
            does not estimate a Hessian) and start_hazard, if present, is
            floored to an int.
    """
    varbound = np.array([param_ranges[k] for k in param_ranges.keys()])
    algorithm_param = {
        "max_num_iteration": 200,
        "population_size": 100,
        "mutation_probability": 0.4,
        "elit_ratio": 0.05,
        "crossover_probability": 0.5,
        "parents_portion": 0.5,
        "crossover_type": "uniform",
        "max_iteration_without_improv": None,
    }

    model = ga(
        function=cost_function,
        dimension=len(param_ranges.keys()),
        convergence_curve=False,
        progress_bar=False,
        variable_type="real",
        variable_boundaries=varbound,
        algorithm_parameters=algorithm_param,
    )

    model.run()

    best_params = model.output_dict["variable"]
    if is_minimzer:
        return refine_with_minimizer(best_params, param_ranges, cost_function)
    else:
        best_fitness = model.output_dict["function"]
        best_params_dict = {k: v for k, v in zip(param_ranges.keys(), best_params)}
        if "start_hazard" in best_params_dict.keys():
            best_params_dict["start_hazard"] = int(
                np.floor(best_params_dict["start_hazard"])
            )
        return (
            best_params_dict,
            best_fitness,
            np.zeros((len(param_ranges.keys()), len(param_ranges.keys()))),
        )


def ga_fit_pygad(param_ranges: dict, cost_function, is_minimzer: bool = False) -> tuple:
    """Fit parameters by minimizing cost_function with pygad's genetic algorithm,
    optionally refined with L-BFGS-B.

    Args:
        param_ranges (dict): Maps parameter name to a (low, high) bound tuple.
            Order determines the parameter vector order passed to cost_function.
        cost_function (Callable[[np.ndarray], float]): Function to minimize.
        is_minimzer (bool, optional): If True, refine the GA's best solution
            with refine_with_minimizer and return its (params, fitness,
            Hessian) instead. Defaults to False.

    Returns:
        tuple: (best_params_dict, best_fitness, hessian_placeholder), where
            hessian_placeholder is a 5x5 zero matrix when is_minimzer is
            False (not an actual Hessian estimate).

    Note: this calls `make_fitness_func_from_cost`, which is not defined or
    imported anywhere in this codebase, so calling this function currently
    raises NameError.
    """
    gene_space = [param_ranges[k] for k in param_ranges.keys()]
    fitness_func = make_fitness_func_from_cost(cost_function)

    random_mutation_min_val = [0.0, 0.01, 0.0, 0.0, -3.0]
    random_mutation_max_val = [1.0, 1.0, 25.0, 2.0, 0.0]

    ga_instance = pygad.GA(
        num_generations=400,
        num_parents_mating=50,
        fitness_func=fitness_func,
        sol_per_pop=100,
        num_genes=len(gene_space),
        gene_space=gene_space,
        mutation_percent_genes=40,
        mutation_type="random",
        mutation_by_replacement=True,
        # random_mutation_min_val=random_mutation_min_val,
        # random_mutation_max_val=random_mutation_max_val,
        gene_type=float,
    )

    ga_instance.run()
    best_solution, best_fitness, _ = ga_instance.best_solution()
    if is_minimzer:
        return refine_with_minimizer(best_solution, param_ranges, cost_function)
    else:
        best_params_dict = {k: v for k, v in zip(param_ranges.keys(), best_solution)}

        return best_params_dict, best_fitness, np.zeros((5, 5))


from scipy.optimize import dual_annealing, differential_evolution


def fit_simulated_annealing(
    param_ranges: dict, cost_function, maxiter: int = 1000, is_minimzer: bool = False
) -> tuple:
    """Fit parameters by minimizing cost_function with scipy's dual annealing,
    optionally refined with L-BFGS-B.

    Args:
        param_ranges (dict): Maps parameter name to a (low, high) bound tuple.
        cost_function (Callable[[np.ndarray], float]): Function to minimize.
        maxiter (int, optional): Max number of dual_annealing iterations.
            Defaults to 1000.
        is_minimzer (bool, optional): If True, refine the annealing result
            with refine_with_minimizer and return its (params, fitness,
            Hessian) instead. Defaults to False.

    Returns:
        tuple: (best_params_dict, best_fitness, hessian_placeholder), where
            hessian_placeholder is a 5x5 zero matrix when is_minimzer is
            False (not an actual Hessian estimate).
    """
    bounds = [param_ranges[k] for k in param_ranges.keys()]

    result = dual_annealing(cost_function, bounds, maxiter=maxiter)
    best_params = result.x
    best_fitness = result.fun
    best_params_dict = {k: v for k, v in zip(param_ranges.keys(), best_params)}
    if is_minimzer:
        return refine_with_minimizer(best_params, param_ranges, cost_function)
    else:

        return best_params_dict, best_fitness, np.zeros((5, 5))


def fit_differential_evolution(
    param_ranges: dict, cost_function, is_minimzer: bool = True
) -> tuple:
    """Fit parameters by minimizing cost_function with scipy's differential
    evolution, optionally refined with L-BFGS-B.

    Args:
        param_ranges (dict): Maps parameter name to a (low, high) bound tuple.
        cost_function (Callable[[np.ndarray], float]): Function to minimize.
        is_minimzer (bool, optional): If True, refine the DE result with
            refine_with_minimizer and return its (params, fitness, Hessian)
            instead. Defaults to True.

    Returns:
        tuple: (best_params_dict, best_fitness, hessian_placeholder), where
            hessian_placeholder is a 5x5 zero matrix when is_minimzer is
            False (not an actual Hessian estimate). start_hazard, if present,
            is floored to an int.
    """
    bounds = [param_ranges[k] for k in param_ranges.keys()]

    result = differential_evolution(
        cost_function,
        bounds,
        strategy="rand1bin",
        maxiter=3000,
        popsize=30,  # Increased population size
        mutation=(0.7, 1.2),  # Higher and wider mutation range
        recombination=0.9,  # Higher recombination
        polish=True,
    )
    best_params = result.x
    best_fitness = result.fun
    best_params_dict = {k: v for k, v in zip(param_ranges.keys(), best_params)}
    if "start_hazard" in best_params_dict.keys():
        best_params_dict["start_hazard"] = int(
            np.floor(best_params_dict["start_hazard"])
        )
    if is_minimzer:
        return refine_with_minimizer(best_params, param_ranges, cost_function)
    else:

        return best_params_dict, best_fitness, np.zeros((5, 5))


# # return best_params_dict, best_fitness
