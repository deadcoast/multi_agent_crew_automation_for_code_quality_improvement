import numpy as np
from bayes_opt import BayesianOptimization

# Define an objective function (e.g., optimizing x^2 * sin(x))
def objective_function(x):
    return -(x**2 * np.sin(x))  # We aim to maximize, so we negate it

# Define parameter bounds
param_bounds = {'x': (-5, 5)}

# Initialize Bayesian Optimizer
optimizer = BayesianOptimization(
    f=objective_function,  # Function to optimize
    pbounds=param_bounds,  # Parameter space
    random_state=42
)

# Run Optimization
optimizer.maximize(init_points=5, n_iter=20)

# Best Parameters Found
print("Best parameters:", optimizer.max)