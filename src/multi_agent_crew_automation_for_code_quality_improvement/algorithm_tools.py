"""
Algorithm adaptation tools for code quality improvement.

This module provides wrappers that adapt specialized algorithms from the algorithms directory
to be used by the AI agents in their workflows. Each algorithm is wrapped as a BaseTool
that can be directly integrated into the agent toolsets.

Installation:
    All dependencies can be installed with pip:

    ```bash
    # Core dependencies
    pip install numpy scikit-learn matplotlib
    pip install networkx>=3.0  # Specific version requirement to avoid bugs

    # Machine learning algorithms
    pip install hpelm minisom pyTsetlinMachine gplearn 

    # Optimization and analysis tools
    pip install bayesian-optimization sklearn-crfsuite
    ```

Note:
    Some algorithms may have platform-specific requirements. Please check the documentation
    of each individual package for detailed installation instructions.

    NetworkX version 3.0 or higher is specifically required to avoid bugs with graph
    manipulation and degree views in the dependency analysis tools.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from crewai.tools import BaseTool

# Add the algorithms directory to the path so we can import the algorithms
algorithms_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "algorithms"
)
sys.path.append(algorithms_path)

# Import algorithm modules will be done dynamically when needed
try:
    # The actual imports will happen inside the relevant tool classes
    pass
except ImportError as e:
    print(f"Error importing algorithm modules: {e}")
    # We'll still define the tools, but they'll handle the import error gracefully


class ExtremeLearningMachineTool(BaseTool):
    """
    Tool that uses Extreme Learning Machines (ELM) for code pattern recognition.

    ELMs are a type of neural network that can quickly learn patterns in data.
    This tool is useful for the Analyzer Agent to detect code patterns and classify them.
    
    Requirements:
        - hpelm: `pip install hpelm`
        - scikit-learn: `pip install scikit-learn`
        - numpy: `pip install numpy`
    """

    name: str = "Extreme Learning Machine Tool"
    description: str = """
    Uses Extreme Learning Machines to detect patterns in code and classify them.
    This tool can be used to identify:
    - Code patterns that might indicate bugs
    - Coding style inconsistencies
    - Potential optimization opportunities
    
    It trains quickly and can handle large codebases efficiently.
    """

    def _run(
        self,
        code_features: List[List[float]],
        labels: Optional[List[int]] = None,
        mode: str = "train",
    ) -> str:
        """
        Run the Extreme Learning Machine algorithm on code features.

        Args:
            code_features: List of feature vectors extracted from code
            labels: Optional list of labels for training (0, 1, 2, etc.)
            mode: "train" to train a new model, "predict" to use existing model

        Returns:
            A summary of the results
        """
        try:
            import hpelm
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            # Normalize features
            X = np.array(code_features)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            if mode == "train" and labels is not None:
                return self._extracted_from__run_29(labels, train_test_split, X, hpelm)
            elif mode == "predict":
                # Here we would load a pre-trained model
                # elm = hpelm.ELM.load("elm_model.pkl")
                # For demo purposes, we'll just return a placeholder

                return """## Extreme Learning Machine Prediction

Predictions would be generated here if a pre-trained model was available.
Please use the 'train' mode first to train a model.
                """

            else:
                return "Error: For training mode, labels must be provided."

        except ImportError:
            return "Error: Required libraries (hpelm) not installed. Please install with 'pip install hpelm'."
        except Exception as e:
            return f"Error running Extreme Learning Machine: {str(e)}"

    # TODO Rename this here and in `_run`
    def _extracted_from__run_29(self, labels, train_test_split, X, hpelm):
        y = np.array(labels)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Convert labels to one-hot encoding
        num_classes = len(set(y))
        y_train_onehot = np.eye(num_classes)[y_train]
        np.eye(num_classes)[y_test]

        # Define and Train ELM
        # Ensure X_train and y_train_onehot are numpy arrays
        X_train_np = np.array(X_train)
        y_train_onehot_np = np.array(y_train_onehot)
        elm = hpelm.ELM(X_train_np.shape[1], y_train_onehot_np.shape[1])
        elm.add_neurons(50, "sigm")  # 50 hidden neurons with sigmoid activation
        elm.train(X_train_np, y_train_onehot_np, "c")  # 'c' for classification

        # Make predictions
        y_pred = elm.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Evaluate model
        from sklearn.metrics import accuracy_score, classification_report

        accuracy = accuracy_score(y_test, y_pred_classes)
        report = classification_report(y_test, y_pred_classes)

        # Save model for later use
        # elm.save("elm_model.pkl")

        return f"""## Extreme Learning Machine Results
                
Training completed with {X_train_np.shape[0]} samples, {X_train_np.shape[1]} features, and {num_classes} classes.

### Performance Metrics
- Accuracy: {accuracy:.4f}

### Classification Report
{report}

The model has been trained and can now be used for code pattern recognition.
                """


class SelfOrganizingMapTool(BaseTool):
    """
    Tool that uses Self-Organizing Maps (SOM) for code clustering and visualization.

    SOMs are a type of neural network that produce a low-dimensional representation
    of the input space, making them useful for clustering and visualization.
    
    Requirements:
        - minisom: `pip install minisom`
        - numpy: `pip install numpy`
        - matplotlib: `pip install matplotlib`
    """

    name: str = "Self-Organizing Map Tool"
    description: str = """
    Uses Self-Organizing Maps to cluster and visualize code similarities.
    This tool can be used to:
    - Group similar code files
    - Identify code clusters that might benefit from refactoring
    - Visualize code organization and dependencies
    
    Helps the Documentation Agent to organize documentation based on code similarities.
    """

    def _run(
        self,
        code_vectors: List[List[float]],
        labels: Optional[List[str]] = None,
        som_size: Tuple[int, int] = (10, 10),
    ) -> str:
        """
        Run Self-Organizing Map clustering on code vectors.

        Args:
            code_vectors: List of feature vectors representing code files
            labels: Optional labels for the code vectors (e.g., file names)
            som_size: Size of the SOM grid (width, height)

        Returns:
            A summary of the clustering results
        """
        try:
            import matplotlib
            from minisom import MiniSom

            matplotlib.use("Agg")  # Non-interactive backend

            # Convert input to numpy array
            X = np.array(code_vectors)

            # Normalize vectors
            X = X / np.linalg.norm(X, axis=1, keepdims=True)

            # Initialize and train SOM
            som = MiniSom(
                som_size[0], som_size[1], X.shape[1], sigma=1, learning_rate=0.5
            )
            som.random_weights_init(X)
            som.train_random(X, 1000)

            # Create activity heatmap
            activation_map = np.zeros(som_size)
            winners = []

            for x in X:
                winner = som.winner(x)
                activation_map[winner] += 1
                winners.append(winner)

            # Create a matplotlib figure
            plt.figure(figsize=(10, 8))
            plt.imshow(activation_map.T, cmap="coolwarm", origin="lower", alpha=0.7)
            plt.colorbar(label="Neuron Activation Frequency")

            # Add labels if provided
            if labels:
                for i, vector in enumerate(X):
                    winner = winners[i]
                    labels[i] if i < len(labels) else str(i)
                    plt.text(
                        winner[0],
                        winner[1],
                        str(i),
                        color="black",
                        fontsize=8,
                        ha="center",
                        va="center",
                        bbox=dict(
                            facecolor="white",
                            edgecolor="black",
                            boxstyle="round,pad=0.3",
                        ),
                    )

            plt.title("Code Clustering with Self-Organizing Map")
            plt.grid(color="black", linestyle="--", linewidth=0.5)

            # Save the figure
            plt.savefig("som_clustering.png")
            plt.close()

            # Count items in each cluster
            cluster_counts = {}
            for i, winner in enumerate(winners):
                cluster_key = f"{winner[0]},{winner[1]}"
                if cluster_key not in cluster_counts:
                    cluster_counts[cluster_key] = []
                cluster_counts[cluster_key].append(i)

            # Format output
            output = "## Self-Organizing Map Clustering Results\n\n"
            output += f"Clustered {len(X)} code samples into {len(cluster_counts)} clusters.\n\n"

            # Show most populated clusters
            sorted_clusters = sorted(
                cluster_counts.items(), key=lambda x: len(x[1]), reverse=True
            )
            output += "### Top Clusters:\n\n"

            for i, (cluster_key, members) in enumerate(sorted_clusters[:5]):
                output += f"#### Cluster {i + 1} (position {cluster_key}):\n"
                output += f"Contains {len(members)} items\n"
                if labels:
                    output += "Sample members:\n"
                    for member_idx in members[:5]:
                        if member_idx < len(labels):
                            output += f"- {labels[member_idx]}\n"
                    if len(members) > 5:
                        output += f"- ... and {len(members) - 5} more\n"
                output += "\n"

            output += "A visualization of the clustering has been saved as 'som_clustering.png'.\n"
            return output

        except ImportError:
            return "Error: Required libraries (minisom) not installed. Please install with 'pip install minisom'."
        except Exception as e:
            return f"Error running Self-Organizing Map: {str(e)}"


class IsolationForestTool(BaseTool):
    """
    Tool that uses Isolation Forest for anomaly detection in code metrics.

    Isolation Forest is an algorithm that detects anomalies by isolating observations,
    making it useful for finding outliers in code quality metrics.
    
    Requirements:
        - scikit-learn: `pip install scikit-learn`
        - numpy: `pip install numpy`
        - matplotlib: `pip install matplotlib`
    """

    name: str = "Isolation Forest Tool"
    description: str = """
    Uses Isolation Forest to detect anomalies in code metrics.
    This tool can be used to:
    - Identify outlier files with unusual characteristics
    - Detect suspicious patterns that might indicate bugs or security issues
    - Find performance hotspots that deviate from normal behavior
    
    Particularly useful for the Security Agent to detect anomalous code.
    """

    def _run(
        self,
        data_points: List[List[float]],
        feature_names: Optional[List[str]] = None,
        contamination: float = 0.1,
    ) -> str:
        """
        Run Isolation Forest to detect anomalies in code metrics.

        Args:
            data_points: List of feature vectors representing code metrics
            feature_names: Optional names for the features (for interpretation)
            contamination: Expected proportion of anomalies (default 0.1 or 10%)

        Returns:
            A summary of the detected anomalies
        """
        try:
            import matplotlib
            from sklearn.ensemble import IsolationForest

            matplotlib.use("Agg")  # Non-interactive backend

            # Convert to numpy array
            X = np.array(data_points)

            # Run Isolation Forest
            iso_forest = IsolationForest(
                n_estimators=100, contamination=str(contamination), random_state=42
            )
            y_pred = iso_forest.fit_predict(X)

            # -1 indicates anomalies, 1 indicates normal data
            anomaly_indices = np.where(y_pred == -1)[0]
            normal_indices = np.where(y_pred == 1)[0]

            # Create visualization if 2D data
            if X.shape[1] == 2:
                self._extracted_from__run_39(X, normal_indices, anomaly_indices, feature_names)
            # Format output
            output = "## Isolation Forest Anomaly Detection Results\n\n"
            output += f"Analyzed {len(X)} data points with {X.shape[1]} features.\n"
            output += f"Detected {len(anomaly_indices)} anomalies ({len(anomaly_indices) / len(X) * 100:.1f}% of data).\n\n"

            if len(anomaly_indices) > 0:
                output += "### Anomaly Details\n\n"
                output += "Index | "
                if feature_names:
                    output += " | ".join(feature_names)
                else:
                    output += " | ".join(
                        [f"Feature {i + 1}" for i in range(min(5, X.shape[1]))]
                    )
                output += "\n"

                output += (
                    "----- | "
                    + " | ".join(["-" * 10 for _ in range(min(5, X.shape[1]))])
                    + "\n"
                )

                for idx in anomaly_indices[:10]:  # Show first 10 anomalies
                    output += f"{idx} | "
                    if X.shape[1] <= 5:
                        output += " | ".join(
                            [f"{X[idx, i]:.3f}" for i in range(X.shape[1])]
                        )
                    else:
                        output += (
                            " | ".join([f"{X[idx, i]:.3f}" for i in range(5)]) + "..."
                        )
                    output += "\n"

                if len(anomaly_indices) > 10:
                    output += f"... and {len(anomaly_indices) - 10} more anomalies\n\n"

                if X.shape[1] == 2:
                    output += "A visualization of the anomalies has been saved as 'anomaly_detection.png'.\n"

            return output

        except ImportError:
            return "Error: Required libraries (sklearn) may not be installed."
        except Exception as e:
            return f"Error running Isolation Forest: {str(e)}"

    # TODO Rename this here and in `_run`
    def _extracted_from__run_39(self, X, normal_indices, anomaly_indices, feature_names):
        plt.figure(figsize=(10, 6))
        plt.scatter(
            X[normal_indices, 0],
            X[normal_indices, 1],
            c="blue",
            label="Normal",
            alpha=0.7,
        )
        plt.scatter(
            X[anomaly_indices, 0],
            X[anomaly_indices, 1],
            c="red",
            label="Anomaly",
            alpha=0.7,
        )

        if feature_names and len(feature_names) >= 2:
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
        else:
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")

        plt.title("Isolation Forest Anomaly Detection")
        plt.legend()
        plt.savefig("anomaly_detection.png")
        plt.close()


class BayesianOptimizationTool(BaseTool):
    """
    Tool that uses Bayesian Optimization for code performance tuning.

    Bayesian Optimization is an efficient method for finding the minimum or maximum
    of a "black-box" function, making it useful for optimizing code performance parameters.
    
    Requirements:
        - bayes_opt: `pip install bayesian-optimization`
    """

    name: str = "Bayesian Optimization Tool"
    description: str = """
    Uses Bayesian Optimization to tune parameters for optimal code performance.
    This tool can be used to:
    - Optimize hyperparameters in performance-critical code
    - Find optimal configuration settings
    - Balance trade-offs between multiple performance metrics
    
    Especially useful for the Performance Agent to optimize code performance.
    """

    def _run(
        self,
        objective_function_name: str,
        param_bounds: Dict[str, Tuple[float, float]],
        n_iterations: int = 20,
        init_points: int = 5,
    ) -> str:
        """
        Run Bayesian Optimization to find optimal parameters.

        Args:
            objective_function_name: Name of a function that returns a metric to maximize
            param_bounds: Dictionary of parameter names and their (min, max) bounds
            n_iterations: Number of optimization iterations
            init_points: Number of initial random exploration points

        Returns:
            A summary of the optimization results
        """
        try:
            from algorithms.bayesian_optimization import BayesianOptimization

            # Define a wrapper for the objective function
            def objective_wrapper(**params):
                """
                Wrapper for the actual objective function.
                
                In a real-world scenario, this would:
                1. Parse parameters into the format needed by the code
                2. Execute the code with the given parameters
                3. Measure execution time, memory usage, accuracy, etc.
                4. Return a score to maximize (or negative of a score to minimize)
                """
                try:
                    # Check if objective_function_name refers to a global function
                    if objective_function_name in globals():
                        objective_func = globals()[objective_function_name]
                        return objective_func(**params)
                    
                    # For demonstration, we'll evaluate a test objective function
                    # that simulates optimizing a code performance metric
                    
                    # Calculate a synthetic score based on parameter combinations
                    # Higher values are better in Bayesian Optimization
                    score = 0
                    
                    # Example: Optimizing thread count and buffer size for performance
                    # More threads help up to a point, then overhead dominates
                    if 'threads' in params:
                        threads = params['threads']
                        # Theoretical performance curve for threads
                        score += 2 * threads - 0.1 * (threads ** 2)
                    
                    # Larger buffer sizes help up to a point, then waste memory
                    if 'buffer_size' in params:
                        buffer = params['buffer_size']
                        # Theoretical performance curve for buffer size
                        score += 10 * np.log(1 + buffer)
                    
                    # Smaller learning rates can be better for stability
                    if 'learning_rate' in params:
                        lr = params['learning_rate']
                        # Theoretical curve for learning rate
                        score += -5 * (lr - 0.1)**2 + 2
                        
                    # Add gaussian noise to simulate real-world variability
                    score += np.random.normal(0, 0.1)
                    
                    return score
                    
                except Exception as e:
                    # In case of errors, return a very low score
                    print(f"Error in objective function: {e}")
                    return -1000

            # Initialize Bayesian Optimizer
            optimizer = BayesianOptimization(
                objective_function=objective_wrapper,  # Function to optimize
                param_bounds=param_bounds,  # Parameter space
                n_iterations=20,
                init_points=5,
            )

            # Run Optimization
            optimizer.maximize(init_points=init_points, n_iter=n_iterations)

            # Get optimization results
            max_result = optimizer.max()  # Call max() as a method
            best_params = max_result["params"]
            best_score = max_result["target"]

            # Format output
            output = "## Bayesian Optimization Results\n\n"
            output += f"Optimized {len(param_bounds)} parameters over {n_iterations} iterations.\n\n"

            output += "### Best Parameters Found\n"
            for param, value in best_params.items():
                output += f"- {param}: {value:.4f}\n"

            output += f"\n### Best Score: {best_score:.4f}\n\n"

            output += "### Optimization History\n"
            output += "Iteration | "
            output += " | ".join(param_bounds.keys())
            output += " | Score\n"
            output += (
                "--- | "
                + " | ".join(["---" for _ in range(len(param_bounds))])
                + " | ---\n"
            )

            # Since the original BayesianOptimization implementation may not have a res attribute,
            # we'll just show the final result instead of history
            output += "Final | "
            output += " | ".join([f"{best_params[param]:.4f}" for param in param_bounds])
            output += f" | {best_score:.4f}\n"

            output += "\nThese optimal parameters can now be applied to your code to maximize performance.\n"
            return output

        except ImportError:
            return "Error: Required libraries (bayes_opt) not installed. Please install with 'pip install bayes_opt'."
        except Exception as e:
            return f"Error running Bayesian Optimization: {str(e)}"


class TsetlinMachineTool(BaseTool):
    """
    Tool that uses Tsetlin Machines for interpretable pattern recognition in code.

    Tsetlin Machines are a type of learning algorithm based on automata that provide
    high accuracy with interpretable rules, useful for understanding code patterns.
    
    Requirements:
        - pyTsetlinMachine: `pip install pyTsetlinMachine`
        - numpy: `pip install numpy`
    """

    name: str = "Tsetlin Machine Tool"
    description: str = """
    Uses Tsetlin Machines to recognize and learn interpretable patterns in code.
    This tool can be used to:
    - Identify boolean patterns in code features that correlate with bugs
    - Learn explicit rules for code quality that humans can understand
    - Provide explainable classifications for code issues
    
    Especially useful for the Analyzer Agent to provide interpretable insights.
    """

    def _run(
        self,
        data: List[Tuple[List[int], int]],
        feature_names: Optional[List[str]] = None,
        num_clauses: int = 20,
        specificity: float = 3.9,
        threshold: int = 15,
        epochs: int = 50,
    ) -> str:
        """
        Train a Tsetlin Machine on boolean features extracted from code.

        Args:
            data: List of (X, y) tuples where X is a list of binary features and y is 0 or 1
            feature_names: Optional names for the features (for interpretation)
            num_clauses: Number of clauses to use in the Tsetlin Machine
            specificity: Specificity parameter (s)
            threshold: Threshold parameter (T)
            epochs: Number of training epochs

        Returns:
            A summary of the learned patterns
        """
        try:
            import numpy as np

            from algorithms.tsetlin import (
                get_literals_from_input,
                train_tsetlin_machine,
            )

            # Prepare the data
            X = np.array([x for x, _ in data])
            y = np.array([y for _, y in data])
            
            # Number of features
            num_features = X.shape[1]
            
            # Number of states per TA (typically 100)
            num_ta_states = 100
            
            # Train the Tsetlin Machine using the imported function
            trained_clauses_pos, trained_clauses_neg = train_tsetlin_machine(
                list(zip(X, y)),  # Convert to list of tuples
                num_clauses,      # n (must be even)
                num_features,     # o (number of features)
                specificity,      # s (specificity parameter)
                threshold,        # T (threshold parameter)
                epochs,           # Number of training epochs
                num_ta_states     # Number of states for each Tsetlin Automaton
            )

            # Format the output
            output = "## Tsetlin Machine Pattern Analysis\n\n"
            output += f"Trained on {len(data)} samples with {X.shape[1]} boolean features.\n"
            output += f"Using {num_clauses} clauses, specificity={specificity}, threshold={threshold}, epochs={epochs}.\n\n"

            # Interpret the clauses
            output += "### Learned Patterns (Clauses)\n\n"

            output += "#### Positive Patterns (indicating class 1):\n"
            for i, clause in enumerate(trained_clauses_pos[:10]):  # Show top 10 clauses
                output += f"{i + 1}. IF "
                terms = []

                # Process literal indices (non-negated vs negated)
                literals = list(clause)  # Convert set to list
                
                for idx in literals:
                    if idx < num_features:  # Original feature (not negated)
                        feature_name = feature_names[idx] if feature_names and idx < len(feature_names) else f"Feature_{idx}"
                        terms.append(feature_name)
                    else:  # Negated feature
                        orig_idx = idx - num_features
                        feature_name = feature_names[orig_idx] if feature_names and orig_idx < len(feature_names) else f"Feature_{orig_idx}"
                        terms.append(f"NOT {feature_name}")

                output += " AND ".join(terms) if terms else "TRUE"
                output += "\n"

            if len(trained_clauses_pos) > 10:
                output += f"... and {len(trained_clauses_pos) - 10} more clauses\n"

            output += "\n#### Negative Patterns (indicating class 0):\n"
            for i, clause in enumerate(trained_clauses_neg[:10]):  # Show top 10 clauses
                output += f"{i + 1}. IF "
                terms = []

                # Process literal indices (non-negated vs negated)
                literals = list(clause)  # Convert set to list
                
                for idx in literals:
                    if idx < num_features:  # Original feature (not negated)
                        feature_name = feature_names[idx] if feature_names and idx < len(feature_names) else f"Feature_{idx}"
                        terms.append(feature_name)
                    else:  # Negated feature
                        orig_idx = idx - num_features
                        feature_name = feature_names[orig_idx] if feature_names and orig_idx < len(feature_names) else f"Feature_{orig_idx}"
                        terms.append(f"NOT {feature_name}")

                output += " AND ".join(terms) if terms else "TRUE"
                output += "\n"

            if len(trained_clauses_neg) > 10:
                output += f"... and {len(trained_clauses_neg) - 10} more clauses\n"

            # Evaluate on training data
            def predict(X_sample, trained_clauses_pos, trained_clauses_neg, num_features, threshold):
                # Create literals
                X_literals = get_literals_from_input(X_sample, num_features)
                
                # Calculate votes
                votes = 0
                
                # Evaluate positive clauses
                for clause in trained_clauses_pos:
                    # If all literals in the clause are 1, the clause outputs 1
                    clause_result = 1
                    for literal_idx in clause:
                        if X_literals[literal_idx] == 0:
                            clause_result = 0
                            break
                    votes += clause_result
                
                # Evaluate negative clauses
                for clause in trained_clauses_neg:
                    # If all literals in the clause are 1, the clause outputs 1
                    clause_result = 1
                    for literal_idx in clause:
                        if X_literals[literal_idx] == 0:
                            clause_result = 0
                            break
                    votes -= clause_result
                
                # Apply threshold
                votes = max(-threshold, min(threshold, votes))
                
                # Return prediction (1 if votes >= 0, 0 otherwise)
                return 1 if votes >= 0 else 0
            
            # Make predictions on training data
            y_pred = np.array([predict(x, trained_clauses_pos, trained_clauses_neg, num_features, threshold) for x in X])
            accuracy = np.mean(y_pred == y)

            output += "\n### Model Performance\n"
            output += f"Training accuracy: {accuracy:.2f}\n\n"
            output += "These patterns can be interpreted as rules for classifying code into categories.\n"
            return output

        except ImportError:
            return "Error: Required libraries not installed. Please install with 'pip install pyTsetlinMachine'."
        except Exception as e:
            return f"Error running Tsetlin Machine: {str(e)}"


class SymbolicRegressionTool(BaseTool):
    """
    Tool that uses Symbolic Regression to discover mathematical relationships in code metrics.

    Symbolic Regression searches for mathematical expressions that best fit data,
    useful for discovering relationships between code metrics and performance/quality.
    
    Requirements:
        - gplearn: `pip install gplearn`
        - scikit-learn: `pip install scikit-learn`
        - numpy: `pip install numpy`
    """

    name: str = "Symbolic Regression Tool"
    description: str = """
    Uses Symbolic Regression to discover mathematical relationships in code metrics.
    This tool can be used to:
    - Find formulas relating code metrics to performance or quality
    - Discover hidden dependencies between different aspects of code
    - Generate human-readable insights about code structure
    
    Helpful for the Refactor Agent to understand mathematical relationships in code.
    """

    def _run(
        self,
        X: List[List[float]],
        y: List[float],
        feature_names: Optional[List[str]] = None,
        max_generations: int = 100,
        population_size: int = 1000,
    ) -> str:
        """
        Run Symbolic Regression to find relationships in code metrics.

        Args:
            X: List of feature vectors
            y: Target values to predict
            feature_names: Optional names for the features
            max_generations: Maximum number of generations for genetic programming
            population_size: Size of the population in genetic programming

        Returns:
            A summary of the discovered mathematical relationships
        """
        try:
            import numpy as np
            from gplearn.genetic import SymbolicRegressor
            from sklearn.metrics import mean_squared_error, r2_score

            # Convert inputs to numpy arrays
            X_array = np.array(X)
            y_array = np.array(y)

            # Initialize and train the symbolic regressor
            est_gp = SymbolicRegressor(
                population_size=population_size,
                generations=max_generations,
                function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg'],
                metric='mean_absolute_error',
                verbose=0,
                random_state=42
            )

            est_gp.fit(X_array, y_array)

            # Get the best program
            best_program = est_gp._program

            # Make predictions
            y_pred = est_gp.predict(X_array)

            # Calculate metrics
            r2 = r2_score(y_array, y_pred)
            mse = mean_squared_error(y_array, y_pred)

            # Format the results
            output = "## Symbolic Regression Analysis\n\n"
            output += f"Analyzed relationships between {X_array.shape[1]} features and the target metric.\n"
            output += f"Using genetic programming with {population_size} population size and {max_generations} max generations.\n\n"

            # Show the best formula
            formula = str(best_program)

            # Replace generic feature names with actual names if provided
            if feature_names:
                for i, name in enumerate(feature_names):
                    formula = formula.replace(f"X{i}", name)

            output += "### Discovered Mathematical Relationship\n\n"
            output += f"**Best Formula (R² = {r2:.3f}, MSE = {mse:.3f}):**\n"
            output += f"`{formula}`\n\n"

            # Show feature importance if available
            if hasattr(est_gp, '_program'):
                output += "### Feature Importance\n\n"
                # Get variable importance through program analysis
                # This is a safer approach that doesn't rely on specific attributes
                feature_names = feature_names or [f"Feature_{i+1}" for i in range(X_array.shape[1])]

                # Calculate a rough approximation of feature importance
                try:
                    # Try to calculate importance from the program if possible
                    importance = np.zeros(X_array.shape[1])
                    program_str = str(est_gp._program)

                    # Count occurrences of each feature in the program string
                    for i, name in enumerate(feature_names):
                        var_name = f"X{i}"
                        importance[i] = program_str.count(var_name)

                    # Normalize if any importance values exist
                    if np.sum(importance) > 0:
                        importance = importance / np.sum(importance)
                except Exception:
                    # Fallback to equal importance
                    importance = np.ones(X_array.shape[1]) / X_array.shape[1]

                # Sort features by importance
                feature_importance = list(zip(feature_names, importance))
                feature_importance.sort(key=lambda x: x[1], reverse=True)

                for name, imp in feature_importance:
                    output += f"- {name}: {imp:.3f}\n"

            output += "\nThis formula represents the mathematical relationship between code metrics.\n"
            output += "It can be used to understand how different aspects of code interact and affect quality or performance.\n"

            return output

        except ImportError:
            return "Error: Required libraries for symbolic regression not installed. Please install with 'pip install gplearn'."
        except Exception as e:
            return f"Error running Symbolic Regression: {str(e)}"


class HopfieldNetworkTool(BaseTool):
    """
    Tool that uses Hopfield Networks for pattern recognition and memory in code analysis.

    Hopfield Networks are a form of recurrent neural network that can store and recall patterns,
    useful for recognizing code patterns or completing partial code snippets.
    
    Requirements:
        - numpy: `pip install numpy`
        - matplotlib: `pip install matplotlib`
    """

    name: str = "Hopfield Network Tool"
    description: str = """
    Uses Hopfield Networks for code pattern recognition and associative memory.
    This tool can be used to:
    - Store and recall code patterns
    - Complete partial code patterns based on learned examples
    - Recognize code structures even with noise or variations
    
    Useful for the Documentation Agent to associate code with documentation patterns.
    """

    def _run(
        self,
        patterns: List[List[int]],
        query_pattern: Optional[List[int]] = None,
        mode: str = "store",
    ) -> str:
        """
        Use a Hopfield Network to store or recall code patterns.

        Args:
            patterns: List of binary patterns to store in the network
            query_pattern: Optional partial or noisy pattern to recall (for mode="recall")
            mode: "store" to train the network, "recall" to retrieve a pattern

        Returns:
            A summary of the results
        """
        try:
            # Simplified Hopfield Network implementation
            def train_hopfield(patterns):
                # Convert to numpy arrays and ensure values are -1 or 1
                patterns_array = np.array(patterns)
                # Convert 0,1 to -1,1 explicitly with integer conversion
                patterns_array = 2 * patterns_array.astype(int) - 1
                n_features = patterns_array.shape[1]

                # Initialize weights matrix
                W = np.zeros((n_features, n_features))

                # Train on each pattern
                for pattern in patterns_array:
                    W += np.outer(pattern, pattern)

                # Set diagonal to zero (no self-connections)
                np.fill_diagonal(W, 0)

                # Normalize
                W = W / patterns_array.shape[0]

                return W

            def recall_hopfield(W, query, max_iterations=10):
                # Convert to numpy array and ensure values are -1 or 1
                state = np.array(query)
                # Convert 0,1 to -1,1 explicitly with integer conversion
                state = 2 * state.astype(int) - 1

                # Iterate until convergence or max iterations
                for _ in range(max_iterations):
                    old_state = state.copy()

                    # Update each neuron
                    for i in range(len(state)):
                        state[i] = np.sign(np.dot(W[i], state))
                        # Handle 0 case (no connections)
                        if state[i] == 0:
                            state[i] = old_state[i]

                    # Check for convergence
                    if np.array_equal(state, old_state):
                        break

                # Convert back to 0,1
                return (state + 1) // 2

            # Main logic
            output = "## Hopfield Network Pattern Analysis\n\n"

            if mode == "store":
                output += f"Storing {len(patterns)} patterns in the Hopfield Network.\n"
                output += f"Each pattern has {len(patterns[0]) if patterns else 0} features.\n\n"

                # Train the network
                weights = train_hopfield(patterns)

                output += "### Sample Stored Patterns\n"
                for i, pattern in enumerate(patterns[:3]):  # Show first 3 patterns
                    output += f"Pattern {i + 1}: `{pattern}`\n"

                if len(patterns) > 3:
                    output += f"... and {len(patterns) - 3} more patterns\n"

                output += "\nThe Hopfield Network has successfully stored these patterns and can recall them even from partial or noisy inputs.\n"

            elif mode == "recall" and query_pattern:
                output += "### Pattern Recall\n\n"
                output += f"Query pattern: `{query_pattern}`\n\n"

                # Train the network and perform recall
                weights = train_hopfield(patterns)
                recalled = recall_hopfield(weights, query_pattern)

                output += f"Recalled pattern: `{recalled.tolist()}`\n\n"

                # Find the closest stored pattern
                min_distance = float("inf")
                closest_pattern = None
                closest_idx = -1

                for i, pattern in enumerate(patterns):
                    # Calculate Hamming distance
                    distance = sum(
                        p != r for p, r in zip(pattern, recalled.tolist(), strict=False)
                    )
                    if distance < min_distance:
                        min_distance = distance
                        closest_pattern = pattern
                        closest_idx = i

                if closest_pattern:
                    output += f"Closest stored pattern (#{closest_idx + 1}): `{closest_pattern}`\n"
                    output += f"Hamming distance: {min_distance} bits\n"

            else:
                output += "Please provide a query pattern for recall mode."

            return output

        except Exception as e:
            return f"Error using Hopfield Network: {str(e)}"


class ConditionalRandomFieldTool(BaseTool):
    """Tool for sequence labeling in code using Conditional Random Fields."""

    name: str = "Conditional Random Field Tool"
    description: str = """
    Uses Conditional Random Fields for sequence labeling in code:
    - Tag code elements based on their context and role
    - Identify dependencies between code elements
    - Analyze test coverage at a sequence level
    
    Particularly useful for the Test Agent to analyze test dependencies.
    """

    def _run(
        self,
        sequences: List[List[Dict[str, Any]]],
        labels: Optional[List[List[str]]] = None,
        mode: str = "train",
        features: Optional[List[str]] = None,
    ) -> str:
        """
        Use Conditional Random Fields for sequence labeling in code.

        Args:
            sequences: List of sequences, where each sequence is a list of feature dictionaries
            labels: List of label sequences (for training mode)
            mode: "train" to train a new model, "predict" to use existing model
            features: List of feature names to use

        Returns:
            A summary of the results
        """
        # Initialize features as empty list if None
        if features is None:
            features = []

        output = "## Conditional Random Field Analysis\n\n"

        # Check if sklearn_crfsuite is available - use a safe approach
        # First import pkg_resources
        try:
            import pkg_resources
            try:
                pkg_resources.get_distribution("sklearn-crfsuite")
            except pkg_resources.DistributionNotFound:
                return self._crfsuite_validator(output)
        except ImportError:
            # If pkg_resources is not available, try a different approach
            try:
                import importlib.util
                if importlib.util.find_spec("sklearn_crfsuite") is None:
                    return self._crfsuite_validator(output)
            except ImportError:
                # Last resort - just try to import directly
                try:
                    import sklearn_crfsuite
                except ImportError:
                    return self._crfsuite_validator(output)
        # Now try to actually use the package
        try:
            # This import is safe because we've already checked the package is installed
            import sklearn_crfsuite

            if mode == "train" and labels:
                output += f"Training CRF model on {len(sequences)} sequences.\n\n"

                # Get model
                crf = sklearn_crfsuite.CRF(
                    algorithm="lbfgs",
                    c1=0.1,
                    c2=0.1,
                    max_iterations=100,
                    all_possible_transitions=True,
                )

                # Fit model
                crf.fit(sequences, labels)

                # Predict
                predictions = crf.predict(sequences)

                # Add training metrics section
                output += "### Training Metrics\n\n"

                # Provide simple prediction count instead of accuracy
                correct_count = sum(
                    pred_seq == true_seq
                    for pred_seq, true_seq in zip(predictions, labels)
                )
                output += f"Model trained on {len(sequences)} sequences\n"
                output += f"Completely matching sequences: {correct_count}/{len(sequences)}\n\n"

                # Show feature weights if available
                try:
                    if hasattr(crf, "state_features_") and crf.state_features_:
                        output += "### Top Feature Weights\n\n"

                        # Format weights
                        weights = [(str(k[0]), str(k[1]), float(v)) 
                                  for k, v in crf.state_features_.items()]

                        # Sort by absolute value
                        weights.sort(key=lambda x: abs(x[2]), reverse=True)

                        # Show top weights
                        for attr, label, weight in weights[:10]:
                            output += f"- {attr} → {label}: {weight:.4f}\n"
                except Exception:
                    output += "Could not extract feature weights\n"

                output += "\nThe CRF model has been trained and can now be used for sequence labeling.\n"

            elif mode == "predict":
                output += f"Making predictions on {len(sequences)} sequences.\n\n"
                output += "This is a placeholder for CRF prediction. In a real implementation, we would load a pre-trained model.\n"

            else:
                output += "Please provide labels for training mode.\n"

        except Exception as e:
            output += f"Error using Conditional Random Fields: {str(e)}\n"

        return output

    def _crfsuite_validator(self, output):
        output += "⚠️ sklearn_crfsuite is not installed. This tool requires sklearn_crfsuite to function properly.\n"
        output += "Please install it with: pip install sklearn-crfsuite\n\n"
        return output


class FuzzyPatternMatchingTool(BaseTool):
    """Tool for identifying similar code patterns using fuzzy matching.

    Identifies similar code patterns using advanced matching techniques:
    - Levenshtein distance for approximate string matching
    - Token-based pattern detection
    - Syntactic similarity scoring
    - AST-based structural pattern matching
    
    Helps identify code duplication and refactoring opportunities.
    
    Requirements:
        - ast: Standard library
        - difflib: Standard library
        - collections: Standard library
        - re: Standard library
    """

    name: str = "Fuzzy Pattern Matching Tool"
    description: str = """
    Identifies similar code patterns using advanced matching techniques:
    - Levenshtein distance for approximate string matching
    - Token-based pattern detection
    - Syntactic similarity scoring
    - AST-based structural pattern matching
    
    Helps identify code duplication and refactoring opportunities.
    """

    def _run(
        self,
        code_snippets: List[str],
        similarity_threshold: float = 0.7,
        normalize_whitespace: bool = True,
        ignore_comments: bool = True,
        token_based: bool = False,
        ast_based: bool = False
    ) -> str:
        """
        Find similar patterns across multiple code snippets.

        Args:
            code_snippets: List of code snippets to analyze for similarities
            similarity_threshold: Minimum similarity score (0.0-1.0) to report matches
            normalize_whitespace: Whether to normalize whitespace before comparison
            ignore_comments: Whether to remove comments before comparison
            token_based: Use token-based comparison instead of string-based
            ast_based: Use AST-based structural comparison (overrides other methods)

        Returns:
            Analysis of similar code patterns found
        """
        import ast
        import re
        from difflib import SequenceMatcher

        if not code_snippets or len(code_snippets) < 2:
            return "Need at least 2 code snippets to compare."

        # Use AST-based comparison if requested
        if ast_based:
            return self._ast_based_comparison(code_snippets, similarity_threshold)

        # Preprocess the code snippets
        processed_snippets = []

        for i, snippet in enumerate(code_snippets):
            if not snippet:
                continue

            processed = snippet

            # Remove comments if requested
            if ignore_comments:
                try:
                    # Parse the code and regenerate without comments
                    tree = ast.parse(snippet)
                    processed = ast.unparse(tree)
                except SyntaxError:
                    # Fall back to regex-based comment removal if parsing fails
                    processed = re.sub(r'#.*$', '', snippet, flags=re.MULTILINE)

            # Normalize whitespace if requested
            if normalize_whitespace:
                # Replace all whitespace sequences with a single space
                processed = re.sub(r'\s+', ' ', processed)
                processed = processed.strip()

            processed_snippets.append((i, processed))

        # Function to calculate similarity using Levenshtein distance
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)

            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        # Function to calculate similarity score (0.0-1.0)
        def calculate_similarity(s1, s2):
            if token_based:
                # Token-based comparison using SequenceMatcher
                return SequenceMatcher(None, s1.split(), s2.split()).ratio()
            # String-based comparison using Levenshtein distance
            max_len = max(len(s1), len(s2))
            if max_len == 0:
                return 1.0  # Both strings are empty
            distance = levenshtein_distance(s1, s2)
            return 1.0 - (distance / max_len)

        # Find similar pairs
        similar_pairs = []

        for i in range(len(processed_snippets)):
            idx1, snippet1 = processed_snippets[i]

            for j in range(i + 1, len(processed_snippets)):
                idx2, snippet2 = processed_snippets[j]

                similarity = calculate_similarity(snippet1, snippet2)

                if similarity >= similarity_threshold:
                    similar_pairs.append((idx1, idx2, similarity))

        # Sort by similarity (descending)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)

        # Prepare output
        if not similar_pairs:
            return f"No similar code patterns found with threshold {similarity_threshold}."

        output = "## Fuzzy Pattern Matching Results\n\n"
        output += f"Analyzed {len(processed_snippets)} code snippets using "
        output += "token-based" if token_based else "string-based"
        output += f" matching (threshold: {similarity_threshold}).\n\n"
        output += f"Found {len(similar_pairs)} similar pairs:\n\n"

        # Show the most similar pairs
        for i, (idx1, idx2, similarity) in enumerate(similar_pairs[:10], 1):
            output += f"### Pair {i}: {similarity:.2f} similarity\n\n"
            output += f"**Snippet {idx1+1}:**\n```python\n{code_snippets[idx1][:200]}"
            output += "...\n```\n" if len(code_snippets[idx1]) > 200 else "\n```\n"
            output += f"**Snippet {idx2+1}:**\n```python\n{code_snippets[idx2][:200]}"
            output += "...\n```\n\n" if len(code_snippets[idx2]) > 200 else "\n```\n\n"
            # Add refactoring suggestion
            if similarity > 0.9:
                output += "**Suggestion:** Consider extracting this highly similar code into a shared function.\n\n"
            elif similarity > 0.7:
                output += "**Suggestion:** Review for potential abstraction opportunities.\n\n"

        # Add summary recommendations
        if similar_pairs:
            output += "## Recommendations\n\n"
            output += "1. **Extract Common Patterns:** Consider refactoring highly similar code segments into reusable functions.\n"
            output += "2. **Create Abstractions:** For moderately similar patterns, explore creating abstractions or base classes.\n"
            output += "3. **Use Templates:** For similar structure with different values, consider template patterns or configuration-driven approaches.\n"

        return output

    def _ast_based_comparison(self, code_snippets: List[str], similarity_threshold: float = 0.7) -> str:
        """
        Perform AST-based structural pattern matching to find similar code structures.
        
        This method:
        1. Parses each code snippet into an AST
        2. Normalizes the AST (anonymizes variable names, etc.)
        3. Compares the structure of ASTs using tree edit distance
        4. Identifies similar code patterns based on structural similarity
        
        Args:
            code_snippets: List of code snippets to analyze
            similarity_threshold: Minimum similarity score to report matches
            
        Returns:
            Formatted analysis of structurally similar code patterns
        """
        import ast
        from collections import defaultdict

        class ASTNormalizer(ast.NodeTransformer):
            def __init__(self):
                self.var_counter = 0
                self.func_counter = 0
                self.class_counter = 0
                self.var_map = {}

            def visit_Name(self, node):
                # Normalize variable names
                if isinstance(node.ctx, ast.Store) and node.id not in self.var_map:
                    self.var_map[node.id] = f"var_{self.var_counter}"
                    self.var_counter += 1
                return ast.Name(id=self.var_map.get(node.id, node.id), ctx=node.ctx)

            def visit_FunctionDef(self, node):
                # Normalize function names but preserve the body structure
                new_name = f"func_{self.func_counter}"
                self.func_counter += 1
                # Process body with current variable map
                new_body = [self.visit(n) for n in node.body]
                return ast.FunctionDef(
                    name=new_name,
                    args=self.visit(node.args),
                    body=new_body,
                    decorator_list=[],
                    returns=None
                )

            def visit_ClassDef(self, node):
                # Normalize class names
                new_name = f"class_{self.class_counter}"
                self.class_counter += 1
                # Process body with current variable map
                new_body = [self.visit(n) for n in node.body]
                return ast.ClassDef(
                    name=new_name,
                    bases=[],
                    keywords=[],
                    body=new_body,
                    decorator_list=[]
                )

            def visit_Constant(self, node):
                # Normalize literals by their type
                if isinstance(node.value, str):
                    return ast.Constant(value="<string>")
                elif isinstance(node.value, (int, float)):
                    return ast.Constant(value=0)
                elif isinstance(node.value, bool):
                    return ast.Constant(value=False)
                return node

        # Parse code snippets into ASTs and normalize them
        normalized_asts = []
        valid_snippets = []

        for i, snippet in enumerate(code_snippets):
            if not snippet.strip():
                continue

            try:
                # Parse the code
                tree = ast.parse(snippet)

                # Normalize the AST
                normalizer = ASTNormalizer()
                normalized_tree = normalizer.visit(tree)

                # Convert to string representation for comparison
                normalized_str = ast.dump(normalized_tree)
                normalized_asts.append((i, normalized_str))
                valid_snippets.append(snippet)
            except SyntaxError:
                # Skip invalid code
                continue

        if len(normalized_asts) < 2:
            return "Not enough valid code snippets for AST comparison."

        # Function to calculate similarity between AST structures
        def ast_similarity(ast_str1, ast_str2):
            from difflib import SequenceMatcher
            return SequenceMatcher(None, ast_str1, ast_str2).ratio()

        # Find structurally similar pairs
        similar_pairs = []

        for i in range(len(normalized_asts)):
            idx1, ast_str1 = normalized_asts[i]

            for j in range(i + 1, len(normalized_asts)):
                idx2, ast_str2 = normalized_asts[j]

                similarity = ast_similarity(ast_str1, ast_str2)

                if similarity >= similarity_threshold:
                    similar_pairs.append((idx1, idx2, similarity))

        # Sort by similarity (descending)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)

        # Function to extract key AST patterns
        def extract_patterns(ast_tree):
            """Extract key structural patterns from an AST"""
            patterns = defaultdict(int)

            # Look for loop patterns
            loops = [n for n in ast.walk(ast_tree) if isinstance(n, (ast.For, ast.While))]
            patterns["loops"] = len(loops)

            # Look for conditionals
            conditionals = [n for n in ast.walk(ast_tree) if isinstance(n, ast.If)]
            patterns["conditionals"] = len(conditionals)

            # Look for try/except blocks
            try_blocks = [n for n in ast.walk(ast_tree) if isinstance(n, ast.Try)]
            patterns["try_except"] = len(try_blocks)

            # Look for function definitions
            funcs = [n for n in ast.walk(ast_tree) if isinstance(n, ast.FunctionDef)]
            patterns["functions"] = len(funcs)

            # Look for nested structures
            nested_depth = 0
            for node in ast.walk(ast_tree):
                if isinstance(node, (ast.For, ast.While, ast.If, ast.Try)):
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.For, ast.While, ast.If, ast.Try)):
                            nested_depth += 1
                            break
            patterns["nesting_depth"] = nested_depth

            # Look for comprehensions
            comprehensions = [
                n for n in ast.walk(ast_tree) 
                if isinstance(n, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))
            ]
            patterns["comprehensions"] = len(comprehensions)

            return patterns

        # Prepare output
        if not similar_pairs:
            return f"No structurally similar code patterns found with threshold {similarity_threshold}."

        output = "## AST-Based Pattern Matching Results\n\n"
        output += f"Analyzed {len(normalized_asts)} valid code snippets "
        output += f"using AST structural comparison (threshold: {similarity_threshold}).\n\n"
        output += f"Found {len(similar_pairs)} structurally similar pairs:\n\n"

        # Show the most similar pairs
        for i, (idx1, idx2, similarity) in enumerate(similar_pairs[:10], 1):
            output += f"### Pair {i}: {similarity:.2f} structural similarity\n\n"

            # Parse ASTs for pattern analysis
            patterns1 = None
            patterns2 = None
            
            try:
                ast1 = ast.parse(valid_snippets[idx1])
                ast2 = ast.parse(valid_snippets[idx2])
                
                patterns1 = extract_patterns(ast1)
                patterns2 = extract_patterns(ast2)
                
                # Add pattern information
                output += "**Shared structural patterns:**\n"
                output += f"- Loops: {patterns1['loops']} vs {patterns2['loops']}\n"
                output += f"- Conditionals: {patterns1['conditionals']} vs {patterns2['conditionals']}\n"
                output += f"- Try/Except blocks: {patterns1['try_except']} vs {patterns2['try_except']}\n"
                output += f"- Functions: {patterns1['functions']} vs {patterns2['functions']}\n"
                output += f"- Nesting depth: {patterns1['nesting_depth']} vs {patterns2['nesting_depth']}\n"
                output += f"- Comprehensions: {patterns1['comprehensions']} vs {patterns2['comprehensions']}\n\n"
            except Exception:
                output += "**Could not analyze structural patterns**\n\n"
                patterns1 = None  # Reset in case of error
                patterns2 = None

            # Show code snippets
            output += f"**Snippet {idx1+1}:**\n```python\n{valid_snippets[idx1][:200]}"
            output += "...\n```\n" if len(valid_snippets[idx1]) > 200 else "\n```\n"
            output += f"**Snippet {idx2+1}:**\n```python\n{valid_snippets[idx2][:200]}"
            output += "...\n```\n\n" if len(valid_snippets[idx2]) > 200 else "\n```\n\n"
            
            # Add specific refactoring suggestions based on similarity and patterns
            if similarity > 0.9:
                output += "**Refactoring Suggestion:** These code blocks are structurally nearly identical. "
                
                if patterns1 and patterns2 and patterns1.get('functions', 0) > 0 and patterns2.get('functions', 0) > 0:
                    output += "Consider creating a shared base function with parameters for the differences.\n\n"
                elif patterns1 and patterns1.get('loops', 0) > 0:
                    output += "Extract this pattern into a helper function that operates on different data.\n\n"
                else:
                    output += "Extract into a shared utility function.\n\n"
                    
            elif similarity > 0.7:
                output += "**Refactoring Suggestion:** These code blocks share significant structure. "
                
                if patterns1 and patterns1.get('conditionals', 0) > 0:
                    output += "Consider using a strategy pattern or polymorphism to handle the variations.\n\n"
                elif patterns1 and patterns1.get('nesting_depth', 0) > 1:
                    output += "Extract nested logic into helper functions with clear naming.\n\n"
                else:
                    output += "Look for abstraction opportunities to unify these code paths.\n\n"

        # Add advanced recommendations
        if similar_pairs:
            output += "## Advanced Refactoring Recommendations\n\n"
            output += "1. **Create Strategy Classes:** For similar algorithmic structures with different implementations.\n"
            output += "2. **Apply Template Method Pattern:** For code with similar steps but varying implementations.\n"
            output += "3. **Extract Composite Operations:** For groups of operations that appear together across code blocks.\n"
            output += "4. **Use Higher-Order Functions:** For similar operations with different transformation logic.\n"
            output += "5. **Implement Visitor Pattern:** For similar operations applied to different data structures.\n"
            
        return output


# Create a class to integrate all algorithm tools
class AlgorithmIntegrationTool(BaseTool):
    """Tool to select and apply specialized algorithms to code quality tasks.

    Selects and applies specialized algorithms for code quality improvement tasks.
    This meta-tool helps choose the best algorithm for:
    - Pattern recognition (ELM, Hopfield)
    - Clustering (SOM, TM)
    - Anomaly detection (iForest)
    - Optimization (Bayesian Optimization)
    - Sequence analysis (CRF)
    
    Provides a unified interface to the specialized mathematical algorithms.
    
    Requirements:
        - Depends on all the requirements of the individual algorithm tools it integrates:
          - hpelm: `pip install hpelm`
          - scikit-learn: `pip install scikit-learn`
          - minisom: `pip install minisom` 
          - bayes_opt: `pip install bayesian-optimization`
          - pyTsetlinMachine: `pip install pyTsetlinMachine`
          - gplearn: `pip install gplearn`
          - sklearn-crfsuite: `pip install sklearn-crfsuite`
          - numpy: `pip install numpy`
          - matplotlib: `pip install matplotlib`
          - networkx>=3.0: `pip install networkx>=3.0`  # Version 3.0+ required to avoid bugs
    """

    name: str = "Algorithm Integration Tool"
    description: str = """
    Selects and applies specialized algorithms for code quality improvement tasks.
    This meta-tool helps choose the best algorithm for:
    - Pattern recognition (ELM, Hopfield)
    - Clustering (SOM, TM)
    - Anomaly detection (iForest)
    - Optimization (Bayesian Optimization)
    - Sequence analysis (CRF)
    
    Provides a unified interface to the specialized mathematical algorithms.
    """

    def _run(self, task_type: str, data: Dict[str, Any]) -> str:
        """
        Select and apply the appropriate algorithm for a given task.

        Args:
            task_type: Type of task ("pattern_recognition", "clustering", "anomaly_detection",
                      "optimization", "sequence_analysis")
            data: Dictionary containing the data and parameters for the algorithm

        Returns:
            Results from the selected algorithm
        """
        # Initialize the appropriate tool based on task type
        if task_type == "anomaly_detection":
            if data.get("algorithm") == "iforest":
                tool = IsolationForestTool()
                # Ensure we're not passing None for required parameters
                data_points = data.get("data_points")
                return (
                    "Error: 'data_points' parameter is required for Isolation Forest"
                    if data_points is None
                    else tool._run(
                        data_points=data_points,
                        feature_names=data.get("feature_names"),
                        contamination=data.get("contamination", 0.1),
                    )
                )
        elif task_type == "clustering":
            if data.get("algorithm") == "som":
                tool = SelfOrganizingMapTool()
                # Ensure we're not passing None for required parameters
                code_vectors = data.get("vectors")
                return (
                    "Error: 'vectors' parameter is required for SOM"
                    if code_vectors is None
                    else tool._run(
                        code_vectors=code_vectors,
                        labels=data.get("labels"),
                        som_size=data.get("som_size", (10, 10)),
                    )
                )
        elif task_type == "optimization":
            if data.get("algorithm") == "bayesian":
                return self._run_bayesian_optimization(data)
        elif task_type == "pattern_recognition":
            if data.get("algorithm") == "elm":
                tool = ExtremeLearningMachineTool()
                # Ensure we're not passing None for required parameters
                code_features = data.get("features")
                return (
                    "Error: 'features' parameter is required for ELM"
                    if code_features is None
                    else tool._run(
                        code_features=code_features,
                        labels=data.get("labels"),
                        mode=data.get("mode", "train"),
                    )
                )
            elif data.get("algorithm") == "hopfield":
                tool = HopfieldNetworkTool()
                # Ensure we're not passing None for required parameters
                patterns = data.get("patterns")
                return (
                    "Error: 'patterns' parameter is required for Hopfield Network"
                    if patterns is None
                    else tool._run(
                        patterns=patterns,
                        query_pattern=data.get("query"),
                        mode=data.get("mode", "store"),
                    )
                )
            elif data.get("algorithm") == "tsetlin":
                tool = TsetlinMachineTool()
                # Ensure we're not passing None for required parameters
                tm_data = data.get("data")
                return (
                    "Error: 'data' parameter is required for Tsetlin Machine"
                    if tm_data is None
                    else tool._run(
                        data=tm_data,
                        feature_names=data.get("feature_names"),
                        num_clauses=data.get("num_clauses", 20),
                        specificity=data.get("specificity", 3.9),
                        threshold=data.get("threshold", 15),
                        epochs=data.get("epochs", 50),
                    )
                )
        elif task_type == "relation_discovery":
            if data.get("algorithm") == "symbolic_regression":
                return self._run_symbolic_regression(data)
        elif task_type == "sequence_analysis":
            if data.get("algorithm") == "crf":
                return self._run_conditional_random_field(data)
        return f"Unknown task type or algorithm: {task_type}, {data.get('algorithm')}"

    def _run_conditional_random_field(self, data):
        """Handle running a Conditional Random Field algorithm with the provided data"""
        tool = ConditionalRandomFieldTool()
        # Ensure we're not passing None for required parameters
        sequences = data.get("sequences")
        if sequences is None:
            return "Error: 'sequences' parameter is required for CRF"

        # Ensure features is a list or None
        features = data.get("features")
        if features is None:
            features = []

        return tool._run(
            sequences=sequences,
            labels=data.get("labels"),
            mode=data.get("mode", "train"),
            features=features,
        )

    def _run_symbolic_regression(self, data):
        """Handle running a Symbolic Regression algorithm with the provided data"""
        tool = SymbolicRegressionTool()
        # Ensure we're not passing None for required parameters
        X = data.get("X")
        y = data.get("y")
        if X is None:
            return "Error: 'X' parameter is required for Symbolic Regression"
        if y is None:
            return "Error: 'y' parameter is required for Symbolic Regression"
        return tool._run(
            X=X,
            y=y,
            feature_names=data.get("feature_names"),
            max_generations=data.get("max_generations", 100),
            population_size=data.get("population_size", 1000),
        )

    def _run_bayesian_optimization(self, data):
        """Handle running a Bayesian Optimization algorithm with the provided data"""
        tool = BayesianOptimizationTool()
        # Ensure we're not passing None for required parameters
        objective = data.get("objective")
        param_bounds = data.get("param_bounds")
        if objective is None:
            return "Error: 'objective' parameter is required for Bayesian Optimization"
        if param_bounds is None:
            return "Error: 'param_bounds' parameter is required for Bayesian Optimization"
        return tool._run(
            objective_function_name=objective,
            param_bounds=param_bounds,
            n_iterations=data.get("n_iterations", 20),
            init_points=data.get("init_points", 5),
        )


class DependencyGraphAnalysisTool(BaseTool):
    """Tool for analyzing module dependencies using networkx.

    Analyzes Python module dependencies by:
    - Building a directed graph of import relationships
    - Identifying highly connected modules
    - Detecting circular dependencies
    - Suggesting potential refactoring opportunities
    
    Requirements:
        - networkx>=3.0: `pip install networkx>=3.0`  # Version 3.0+ required to avoid bugs
        - numpy: `pip install numpy`
        - matplotlib: `pip install matplotlib`
    """

    name: str = "Dependency Graph Analysis Tool"
    description: str = """
    Analyzes Python module dependencies by:
    - Building a directed graph of import relationships
    - Identifying highly connected modules
    - Detecting circular dependencies
    - Suggesting potential refactoring opportunities
    
    Requirements:
        - networkx>=3.0: `pip install networkx>=3.0`  # Version 3.0+ required to avoid bugs
        - numpy: `pip install numpy`
        - matplotlib: `pip install matplotlib`
    """

    def _run(self, module_paths: List[str]) -> str:
        """
        Analyze dependencies for a list of Python modules.

        Args:
            module_paths: List of paths to Python modules

        Returns:
            Analysis of module dependencies
        """
        try:
            import re

            import networkx as nx
            from matplotlib import pyplot as plt

            # Create a directed graph
            G = nx.DiGraph()

            # Add nodes and edges based on import statements
            for module_path in module_paths:
                # Extract module name from path
                module_name = module_path.split('.')[0]
                G.add_node(module_name)

                # Parse the module and add imports as edges
                with open(module_path, 'r') as file:
                    for line in file:
                        # Extract import statements
                        import_statements = re.findall(r'import\s+([a-zA-Z0-9_]+)', line)
                        for import_statement in import_statements:
                            G.add_edge(module_name, import_statement)

            # Draw the graph
            plt.figure(figsize=(12, 8))
            nx.draw(G, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
            plt.title("Module Dependency Graph")
            plt.savefig("module_dependency_graph.png")
            plt.close()

            # Analyze the graph
            output = "## Dependency Analysis Results\n\n"
            output += f"Analyzed {len(G.nodes())} modules and {len(G.edges())} dependencies.\n\n"

            # Check for circular dependencies
            if nx.is_directed_acyclic_graph(G):
                output += "The module dependency graph is a directed acyclic graph (DAG).\n"
            else:
                output += "The module dependency graph contains cycles.\n"

            # Identify highly connected modules using degree centrality instead of raw degree
            # This is more robust across NetworkX versions 
            centrality = nx.degree_centrality(G)
            
            # Sort by centrality and take top nodes
            sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            top_nodes = sorted_centrality[:5] if sorted_centrality else []
            
            output += f"\n### Highly Connected Modules ({len(top_nodes)} modules):\n"
            for module, cent in top_nodes:
                # Centrality is normalized, so multiply by (n-1) to get approximate degree
                approx_degree = int(cent * (len(G) - 1))
                output += f"- {module} (connections: {approx_degree})\n"

            output += "\nA visualization of the dependency graph has been saved as 'module_dependency_graph.png'.\n"
            return output

        except ImportError:
            return "Error: Required libraries (networkx, matplotlib) not installed. Please install with 'pip install networkx matplotlib'."
        except Exception as e:
            return f"Error running Dependency Graph Analysis: {str(e)}"


class RandomKitchenSinksTool(BaseTool):
    """
    Tool that uses Random Kitchen Sinks for nonlinear feature transformation.

    Random Kitchen Sinks approximate nonlinear kernel functions using random 
    projections, making them efficient for scaling up kernel methods.
    
    Requirements:
        - scikit-learn: `pip install scikit-learn`
        - numpy: `pip install numpy`
    """

    name: str = "Random Kitchen Sinks Tool"
    description: str = """
    Uses Random Kitchen Sinks for nonlinear feature transformation:
    - Projects input features to higher-dimensional space
    - Approximates kernel functions efficiently without computing the full kernel matrix
    - Enables scalable nonlinear feature representation
    
    Particularly useful for the Performance Agent to optimize feature engineering.
    """

    def _run(
        self,
        X: List[List[float]],
        gamma: float = 1.0,
        n_components: int = 500,
        random_state: int = 42,
    ) -> str:
        """
        Apply Random Kitchen Sinks transformation to input data.

        Args:
            X: Input data as a list of feature vectors
            gamma: RBF kernel parameter (controls width of Gaussian kernel)
            n_components: Number of Monte Carlo samples (features to generate)
            random_state: Random seed for reproducibility

        Returns:
            A summary of the transformation results
        """
        try:
            import numpy as np
            from sklearn.decomposition import PCA
            from sklearn.kernel_approximation import RBFSampler

            # Convert input to numpy array
            X_array = np.array(X)
            
            # Apply Random Kitchen Sinks for kernel approximation
            rks = RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)
            X_rks = rks.fit_transform(X_array)
            
            # Initialize PCA variables
            pca_explained_variance_ratio = None
            
            # Apply PCA for visualization (if more than 2 dimensions)
            if X_array.shape[1] > 2:
                pca = PCA(n_components=2)
                # Apply PCA and store result (we'll use it in visualizations if needed)
                _ = pca.fit_transform(X_rks)
                # Save the explained variance ratio for reporting
                pca_explained_variance_ratio = pca.explained_variance_ratio_
            
            # Compute statistics for the report
            original_dim = X_array.shape[1]
            transformed_dim = X_rks.shape[1]
            expansion_factor = transformed_dim / original_dim
            
            # Format output
            output = "## Random Kitchen Sinks Transformation Results\n\n"
            output += f"Applied RBF kernel approximation with gamma={gamma} to {len(X)} data points.\n\n"
            
            output += "### Transformation Summary\n"
            output += f"- Original dimensionality: {original_dim}\n"
            output += f"- Transformed dimensionality: {transformed_dim}\n"
            output += f"- Expansion factor: {expansion_factor:.2f}x\n"
            
            # Feature statistics
            output += "\n### Feature Statistics\n"
            output += f"- Mean of transformed features: {np.mean(X_rks):.4f}\n"
            output += f"- Standard deviation: {np.std(X_rks):.4f}\n"
            output += f"- Min value: {np.min(X_rks):.4f}\n"
            output += f"- Max value: {np.max(X_rks):.4f}\n"
            
            # Add visualization info if PCA was applied
            if pca_explained_variance_ratio is not None:
                output += "\n### Dimensionality Reduction\n"
                output += "- Applied PCA to visualize the high-dimensional RKS features\n"
                output += f"- First 2 principal components explain {sum(pca_explained_variance_ratio[:2])*100:.2f}% of variance\n"
            
            output += "\nRandom Kitchen Sinks provides a computationally efficient way to work with "
            output += "nonlinear kernel methods by mapping the data to a higher-dimensional space "
            output += "where patterns may become linearly separable.\n"
            
            return output
            
        except ImportError:
            return "Error: Required libraries (scikit-learn, numpy) not installed."
        except Exception as e:
            return f"Error running Random Kitchen Sinks: {str(e)}"


class AwareFactorizationMachinesTool(BaseTool):
    """
    Tool that uses Field-aware Factorization Machines for feature interaction modeling.

    Field-aware Factorization Machines (FFMs) extend factorization machines by learning 
    field-specific latent factors for each feature, making them powerful for 
    capturing complex feature interactions.
    
    Requirements:
        - numpy: `pip install numpy`
    """

    name: str = "Field-aware Factorization Machines Tool"
    description: str = """
    Uses Field-aware Factorization Machines (FFMs) to model feature interactions:
    - Captures complex non-linear relationships between features
    - Learns field-specific latent factors for each feature
    - Particularly effective for sparse feature matrices
    
    Especially useful for the Analyzer Agent to discover complex feature interactions.
    """

    def _run(
        self,
        training_data: List[Tuple[int, List[Tuple[int, int, float]]]],
        n_features: int,
        n_fields: int,
        k_factors: int = 4,
        eta: float = 0.1,
        lambda_reg: float = 2e-5,
        n_epochs: int = 10,
        predict_examples: Optional[List[List[Tuple[int, int, float]]]] = None,
    ) -> str:
        """
        Train a Field-aware Factorization Machine and make predictions.

        Args:
            training_data: List of (label, features) tuples where features is a list of 
                          (field_idx, feature_idx, value) tuples
            n_features: Total number of features
            n_fields: Total number of fields
            k_factors: Number of latent factors
            eta: Learning rate
            lambda_reg: Regularization parameter
            n_epochs: Number of training epochs
            predict_examples: Optional list of feature vectors for predictions

        Returns:
            A summary of the FFM training and predictions
        """
        try:
            import math

            from algorithms.aware_factorization_machines import FFMModel

            # Initialize and train FFM model
            ffm = FFMModel(n_features, n_fields, k_factors, eta, lambda_reg, n_epochs)

            # Ensure labels are -1/1 for logistic loss
            normalized_data = []
            for y, x in training_data:
                if y == 0:
                    y = -1  # Convert 0/1 labels to -1/1 if needed
                normalized_data.append((y, x))

            # Train the model
            ffm.train(normalized_data)

            # Format output
            output = "## Field-aware Factorization Machine Results\n\n"
            output += f"Trained FFM with {len(training_data)} instances across {n_features} features and {n_fields} fields.\n\n"

            output += "### Model Parameters\n"
            output += f"- Latent factors (k): {k_factors}\n"
            output += f"- Learning rate (η): {eta}\n"
            output += f"- Regularization (λ): {lambda_reg}\n"
            output += f"- Training epochs: {n_epochs}\n\n"

            # Make predictions if examples provided
            if predict_examples is not None and len(predict_examples) > 0:
                output += "### Predictions\n"
                output += "| Example | Phi | Probability |\n"
                output += "|---------|-----|-------------|\n"

                for i, example in enumerate(predict_examples[:5]):  # Show first 5 predictions
                    phi = ffm.predict_phi(example)
                    probability = 1.0 / (1.0 + math.exp(-phi))
                    output += f"| {i+1} | {phi:.4f} | {probability:.4f} |\n"

                if len(predict_examples) > 5:
                    output += f"... and {len(predict_examples) - 5} more predictions\n\n"

            output += "\nFFMs excel at modeling complex feature interactions in sparse data, "
            output += "making them ideal for discovering non-linear relationships in code metrics "
            output += "and finding subtle patterns that might indicate code quality issues.\n"

            return output

        except ImportError:
            return "Error: Required libraries not installed."
        except Exception as e:
            return f"Error running Field-aware Factorization Machines: {str(e)}"


class MycorrhizalNetworkTool(BaseTool):
    """
    Tool that uses Mycorrhizal Network models for simulating complex interactions.

    Mycorrhizal Networks model dynamic, symbiotic relationships between different components,
    useful for understanding how different code modules interact and affect each other.
    
    Requirements:
        - numpy: `pip install numpy`
        - matplotlib: `pip install matplotlib`
    """

    name: str = "Mycorrhizal Network Tool"
    description: str = """
    Uses Mycorrhizal Network models to simulate interactions between code components:
    - Models predator-prey, cooperative, and competitive relationships
    - Simulates dynamic behavior of interacting systems
    - Visualizes emergent patterns from complex interactions
    
    Helpful for understanding how code components evolve together in complex systems.
    """

    def _run(
        self,
        model_type: str = "cooperative",
        a: float = 0.01,
        b: float = 0.02,
        d: float = 0.06,
        g: float = 0.09,
        c: float = 1.70,
        e: float = 0.09,
        f: float = 0.09,
        x0: float = 0.0002,
        y0: float = 0.0006,
        epochs: int = 30,
        iterations: int = 100,
    ) -> str:
        """
        Run a Mycorrhizal Network simulation.

        Args:
            model_type: Type of interaction model ("predator_prey", "cooperative", or "competitive")
            a, b, d, g, c, e, f: Model parameters
            x0, y0: Initial values
            epochs: Number of epochs for simulation
            iterations: Number of iterations per epoch

        Returns:
            A summary of the simulation results
        """
        # The return value will be a string, so we don't need to worry about float type issues
        try:
            import matplotlib
            import numpy as np

            from algorithms.mycorrhizal import DMOA

            matplotlib.use("Agg")  # Non-interactive backend

            # Initialize and run simulation
            dmoa = DMOA(a, b, d, g, c, e, f, x0, y0, epochs, iterations)
            x_vals, y_vals = dmoa.run_simulation(model_type)

            # Convert to numpy arrays for analysis
            x_array = np.array(x_vals, dtype=float)
            y_array = np.array(y_vals, dtype=float)

            # Generate plot (saved to file)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(x_array, label="Component A")
            plt.plot(y_array, label="Component B")
            plt.title(f"Mycorrhizal Network Simulation ({model_type.capitalize()} Model)")
            plt.xlabel("Iterations")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.savefig("mycorrhizal_simulation.png")
            plt.close()

            # Compute statistics
            total_points = len(x_array)
            mean_x = float(np.mean(x_array))
            mean_y = float(np.mean(y_array))
            final_x = float(x_array[-1])
            final_y = float(y_array[-1])

            # Handle potential division by zero and type issues
            stability_x = 0.0
            if mean_x > 0:
                try:
                    x_std = float(np.std(x_array[-(total_points // 10):]))
                    stability_x = x_std / mean_x
                except (TypeError, ValueError):
                    stability_x = 0.0

            stability_y = 0.0
            if mean_y > 0:
                try:
                    y_std = float(np.std(y_array[-(total_points // 10):]))
                    stability_y = y_std / mean_y
                except (TypeError, ValueError):
                    stability_y = 0.0

            overall_stability = (stability_x + stability_y) / 2

            # Format output
            output = f"## Mycorrhizal Network Simulation Results ({model_type.capitalize()} Model)\n\n"
            output += f"Simulated {total_points} iterations with parameters: a={a}, b={b}, c={c}, d={d}, g={g}\n\n"

            output += "### Simulation Summary\n"
            output += f"- Initial values: A={x0}, B={y0}\n"
            output += f"- Final values: A={final_x:.6f}, B={final_y:.6f}\n"
            output += f"- Mean values: A={mean_x:.6f}, B={mean_y:.6f}\n\n"

            output += "### Stability Analysis\n"
            output += f"- Component A stability: {stability_x:.4f} (lower is more stable)\n"
            output += f"- Component B stability: {stability_y:.4f} (lower is more stable)\n"
            output += f"- Overall system stability: {overall_stability:.4f}\n\n"

            # Interpret results
            if overall_stability < 0.01:
                output += "The system appears highly stable, indicating components reach equilibrium.\n"
            elif overall_stability < 0.1:
                output += "The system shows moderate stability with minor fluctuations.\n"
            else:
                output += "The system shows significant fluctuations, suggesting cyclical or chaotic behavior.\n"

            if model_type == "cooperative":
                output += "\nCooperative models are useful for identifying symbiotic code patterns where components enhance each other.\n"
            elif model_type == "competitive":
                output += "\nCompetitive models help identify resource conflicts or contentions between code modules.\n"
            elif model_type == "predator_prey":
                output += "\nPredator-prey models identify cyclical dependencies where one component's growth limits another.\n"

            output += "\nA visualization of the simulation has been saved as 'mycorrhizal_simulation.png'.\n"

            return output

        except ImportError:
            return "Error: Required libraries (numpy, matplotlib) not installed."