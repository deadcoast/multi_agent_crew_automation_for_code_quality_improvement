import numpy as np


class BayesianOptimization:
    def __init__(self, objective_function, param_bounds, n_iterations, init_points):
        self.objective_function = objective_function
        self.param_bounds = param_bounds
        self.n_iterations = n_iterations
        self.init_points = init_points
        # Store results and best parameters
        self.res = []
        self._max = {"params": {}, "target": -float("inf")}

    def maximize(self, init_points, n_iter):
        """Run optimization for the given iterations"""
        # Initialize randomly within parameter bounds
        for _ in range(init_points):
            self._sample_and_evaluate()
            
        # Run optimization iterations
        for _ in range(n_iter):
            # For simplicity in this implementation, we'll just do random sampling
            # A real implementation would use Gaussian processes to model the function
            # and pick points that maximize the expected improvement
            self._sample_and_evaluate()
            
        return self._max
    
    def _sample_and_evaluate(self):
        """Sample a random point and evaluate it"""
        # Generate random parameters within bounds
        params = {}
        for param_name, bounds in self.param_bounds.items():
            low, high = bounds
            params[param_name] = np.random.uniform(low, high)
        
        # Evaluate objective function
        try:
            target = self.objective_function(**params)
            # Record result
            result = {"params": params, "target": target}
            self.res.append(result)
            
            # Update maximum if needed
            if target > self._max["target"]:
                self._max = result
                
        except Exception as e:
            print(f"Error evaluating function: {e}")
    
    def max(self):
        """Return the best parameters found"""
        return self._max

    # Example function and parameters (used when this class is used directly)
    def objective_function(self, x):
        return -(x**2 * np.sin(x))  # We aim to maximize, so we negate it

    # Define parameter bounds
    param_bounds = {'x': (-5, 5)}

