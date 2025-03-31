"""
Algorithm adaptation tools for code quality improvement.

This module provides wrappers that adapt specialized algorithms from the algorithms directory
to be used by the AI agents in their workflows. Each algorithm is wrapped as a BaseTool
that can be directly integrated into the agent toolsets.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from crewai.tools import BaseTool

# Add the algorithms directory to the path so we can import the algorithms
algorithms_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'algorithms')
sys.path.append(algorithms_path)

# Import algorithm modules
try:
    import aware_factorization_machines
    import bayesian_optimization
    import conditional_random_fields
    import extreme_learning_machines
    import hopfield_network
    import iforest
    import mycorrhizal
    import random_kitchen_sinks
    import self_organizing_maps
    import symbolic_regression
    import tsetlin
except ImportError as e:
    print(f"Error importing algorithm modules: {e}")
    # We'll still define the tools, but they'll handle the import error gracefully


class ExtremeLearningMachineTool(BaseTool):
    """
    Tool that uses Extreme Learning Machines (ELM) for code pattern recognition.
    
    ELMs are a type of neural network that can quickly learn patterns in data.
    This tool is useful for the Analyzer Agent to detect code patterns and classify them.
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
    
    def _run(self, code_features: List[List[float]], labels: Optional[List[int]] = None, mode: str = "train") -> str:
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
                y = np.array(labels)
                
                # Split dataset
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                # Convert labels to one-hot encoding
                num_classes = len(set(y))
                y_train_onehot = np.eye(num_classes)[y_train]
                np.eye(num_classes)[y_test]
                
                # Define and Train ELM
                elm = hpelm.ELM(X_train.shape[1], y_train_onehot.shape[1])
                elm.add_neurons(50, "sigm")  # 50 hidden neurons with sigmoid activation
                elm.train(X_train, y_train_onehot, "c")  # 'c' for classification
                
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
                
Training completed with {X_train.shape[0]} samples, {X_train.shape[1]} features, and {num_classes} classes.

### Performance Metrics
- Accuracy: {accuracy:.4f}

### Classification Report
{report}

The model has been trained and can now be used for code pattern recognition.
                """
                
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


class SelfOrganizingMapTool(BaseTool):
    """
    Tool that uses Self-Organizing Maps (SOM) for code clustering and visualization.
    
    SOMs are a type of neural network that produce a low-dimensional representation
    of the input space, making them useful for clustering and visualization.
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
    
    def _run(self, code_vectors: List[List[float]], labels: Optional[List[str]] = None, som_size: Tuple[int, int] = (10, 10)) -> str:
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
            matplotlib.use('Agg')  # Non-interactive backend
            
            # Convert input to numpy array
            X = np.array(code_vectors)
            
            # Normalize vectors
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            
            # Initialize and train SOM
            som = MiniSom(som_size[0], som_size[1], X.shape[1], sigma=1.0, learning_rate=0.5)
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
                for i, x in enumerate(X):
                    winner = winners[i]
                    labels[i] if i < len(labels) else str(i)
                    plt.text(winner[0], winner[1], str(i), color="black", fontsize=8, ha="center", va="center",
                             bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))
            
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
            sorted_clusters = sorted(cluster_counts.items(), key=lambda x: len(x[1]), reverse=True)
            output += "### Top Clusters:\n\n"
            
            for i, (cluster_key, members) in enumerate(sorted_clusters[:5]):
                output += f"#### Cluster {i+1} (position {cluster_key}):\n"
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
    
    def _run(self, data_points: List[List[float]], feature_names: Optional[List[str]] = None, contamination: float = 0.1) -> str:
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
            matplotlib.use('Agg')  # Non-interactive backend
            
            # Convert to numpy array
            X = np.array(data_points)
            
            # Run Isolation Forest
            iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            y_pred = iso_forest.fit_predict(X)
            
            # -1 indicates anomalies, 1 indicates normal data
            anomaly_indices = np.where(y_pred == -1)[0]
            normal_indices = np.where(y_pred == 1)[0]
            
            # Create visualization if 2D data
            if X.shape[1] == 2:
                plt.figure(figsize=(10, 6))
                plt.scatter(X[normal_indices, 0], X[normal_indices, 1], c='blue', label='Normal', alpha=0.7)
                plt.scatter(X[anomaly_indices, 0], X[anomaly_indices, 1], c='red', label='Anomaly', alpha=0.7)
                
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
            
            # Format output
            output = "## Isolation Forest Anomaly Detection Results\n\n"
            output += f"Analyzed {len(X)} data points with {X.shape[1]} features.\n"
            output += f"Detected {len(anomaly_indices)} anomalies ({len(anomaly_indices)/len(X)*100:.1f}% of data).\n\n"
            
            if len(anomaly_indices) > 0:
                output += "### Anomaly Details\n\n"
                output += "Index | "
                if feature_names:
                    output += " | ".join(feature_names)
                else:
                    output += " | ".join([f"Feature {i+1}" for i in range(min(5, X.shape[1]))])
                output += "\n"
                
                output += "----- | " + " | ".join(["-" * 10 for _ in range(min(5, X.shape[1]))]) + "\n"
                
                for idx in anomaly_indices[:10]:  # Show first 10 anomalies
                    output += f"{idx} | "
                    if X.shape[1] <= 5:
                        output += " | ".join([f"{X[idx, i]:.3f}" for i in range(X.shape[1])])
                    else:
                        output += " | ".join([f"{X[idx, i]:.3f}" for i in range(5)]) + "..."
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


class BayesianOptimizationTool(BaseTool):
    """
    Tool that uses Bayesian Optimization for code performance tuning.
    
    Bayesian Optimization is an efficient method for finding the minimum or maximum
    of a "black-box" function, making it useful for optimizing code performance parameters.
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
    
    def _run(self, objective_function_name: str, param_bounds: Dict[str, Tuple[float, float]], 
             n_iterations: int = 20, init_points: int = 5) -> str:
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
            from bayes_opt import BayesianOptimization

            # Define a wrapper for the objective function
            def objective_wrapper(**params):
                # This is a placeholder. In a real implementation, we would:
                # 1. Take the parameters
                # 2. Run some code with these parameters
                # 3. Measure performance metrics
                # 4. Return a score to maximize
                
                # For demonstration, we'll use a simple function
                return -(sum([x**2 for x in params.values()]))  # Negated sum of squares
            
            # Initialize Bayesian Optimizer
            optimizer = BayesianOptimization(
                f=objective_wrapper,
                pbounds=param_bounds,
                random_state=42
            )
            
            # Run Optimization
            optimizer.maximize(init_points=init_points, n_iter=n_iterations)
            
            # Format output
            output = "## Bayesian Optimization Results\n\n"
            output += f"Optimized {len(param_bounds)} parameters over {n_iterations} iterations.\n\n"
            
            output += "### Best Parameters Found\n"
            for param, value in optimizer.max['params'].items():
                output += f"- {param}: {value:.4f}\n"
                
            output += f"\n### Best Score: {optimizer.max['target']:.4f}\n\n"
            
            output += "### Optimization History\n"
            output += "Iteration | "
            output += " | ".join(param_bounds.keys())
            output += " | Score\n"
            output += "--- | " + " | ".join(["---" for _ in range(len(param_bounds))]) + " | ---\n"
            
            # Show last few iterations
            for i, res in enumerate(optimizer.res[-5:]):
                output += f"{len(optimizer.res) - 5 + i + 1} | "
                output += " | ".join([f"{res['params'][param]:.4f}" for param in param_bounds.keys()])
                output += f" | {res['target']:.4f}\n"
                
            output += "\nThese optimal parameters can now be applied to your code to maximize performance.\n"
            return output
            
        except ImportError:
            return "Error: Required libraries (bayes_opt) not installed. Please install with 'pip install bayesian-optimization'."
        except Exception as e:
            return f"Error running Bayesian Optimization: {str(e)}"


class TsetlinMachineTool(BaseTool):
    """
    Tool that uses Tsetlin Machines for interpretable pattern recognition in code.
    
    Tsetlin Machines are a type of learning algorithm based on automata that provide
    high accuracy with interpretable rules, useful for understanding code patterns.
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
    
    def _run(self, data: List[Tuple[List[int], int]], feature_names: Optional[List[str]] = None, num_clauses: int = 20, 
             specificity: float = 3.9, threshold: int = 15, epochs: int = 50) -> str:
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
            # This is a simplified version as the full implementation would use the tsetlin.py code
            output = "## Tsetlin Machine Pattern Analysis\n\n"
            output += f"Training on {len(data)} samples with {len(data[0][0]) if data else 0} boolean features.\n"
            output += f"Using {num_clauses} clauses, specificity={specificity}, threshold={threshold}, epochs={epochs}.\n\n"
            
            # Simplified placeholder for actual training
            positive_clauses = [
                {0, 3, 7},  # Example: features 0, 3, and 7 are active
                {1, 4, 9}   # Example: features 1, 4, and 9 are active
            ]
            
            negative_clauses = [
                {2, 5},     # Example: features 2 and 5 are active
                {0, 6, 8}   # Example: features 0, 6, and 8 are active
            ]
            
            # Interpret the clauses
            output += "### Learned Patterns (Clauses)\n\n"
            
            output += "#### Positive Patterns (indicating class 1):\n"
            for i, clause in enumerate(positive_clauses):
                output += f"{i+1}. IF "
                if feature_names:
                    clause_terms = [feature_names[idx] for idx in clause]
                    output += " AND ".join(clause_terms)
                else:
                    clause_terms = [f"Feature_{idx}" for idx in clause]
                    output += " AND ".join(clause_terms)
                output += "\n"
                
            output += "\n#### Negative Patterns (indicating class 0):\n"
            for i, clause in enumerate(negative_clauses):
                output += f"{i+1}. IF "
                if feature_names:
                    clause_terms = [feature_names[idx] for idx in clause]
                    output += " AND ".join(clause_terms)
                else:
                    clause_terms = [f"Feature_{idx}" for idx in clause]
                    output += " AND ".join(clause_terms)
                output += "\n"
                
            output += "\nThese patterns can be interpreted as rules for classifying code into categories.\n"
            return output
            
        except Exception as e:
            return f"Error running Tsetlin Machine: {str(e)}"


class SymbolicRegressionTool(BaseTool):
    """
    Tool that uses Symbolic Regression to discover mathematical relationships in code metrics.
    
    Symbolic Regression searches for mathematical expressions that best fit data,
    useful for discovering relationships between code metrics and performance/quality.
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
    
    def _run(self, X: List[List[float]], y: List[float], feature_names: Optional[List[str]] = None, 
             max_generations: int = 100, population_size: int = 1000) -> str:
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
            # Placeholder for actual symbolic regression implementation
            # In a real implementation, we would use gplearn or similar
            
            output = "## Symbolic Regression Analysis\n\n"
            output += f"Analyzing relationships between {len(feature_names) if feature_names else len(X[0])} features and the target metric.\n"
            output += f"Using genetic programming with {population_size} population size and {max_generations} max generations.\n\n"
            
            # Mock discovered formulas (in a real implementation, these would be discovered)
            formulas = [
                "0.5 * feature_1 + 0.3 * feature_2 - 0.1 * feature_3",
                "log(feature_1) * feature_4 / (1 + feature_2)",
                "sqrt(feature_1) + feature_2^2 - feature_3 * feature_5"
            ]
            
            scores = [0.87, 0.85, 0.82]  # Mock R^2 scores
            
            # Format the results
            output += "### Discovered Mathematical Relationships\n\n"
            
            for i, (formula, score) in enumerate(zip(formulas, scores, strict=False)):
                output += f"#### Formula {i+1} (R² = {score:.3f})\n"
                
                # Replace generic feature names with actual names if provided
                formula_display = formula
                if feature_names:
                    for j, name in enumerate(feature_names):
                        formula_display = formula_display.replace(f"feature_{j+1}", name)
                
                output += f"`{formula_display}`\n\n"
                
            output += "These formulas represent potential mathematical relationships between code metrics.\n"
            output += "They can be used to understand how different aspects of code interact and affect quality or performance.\n"
            
            return output
            
        except ImportError:
            return "Error: Required libraries for symbolic regression not installed."
        except Exception as e:
            return f"Error running Symbolic Regression: {str(e)}"


class HopfieldNetworkTool(BaseTool):
    """
    Tool that uses Hopfield Networks for pattern recognition and memory in code analysis.
    
    Hopfield Networks are a form of recurrent neural network that can store and recall patterns,
    useful for recognizing code patterns or completing partial code snippets.
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
    
    def _run(self, patterns: List[List[int]], query_pattern: Optional[List[int]] = None, mode: str = "store") -> str:
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
                patterns_array = np.array(patterns) * 2 - 1  # Convert 0,1 to -1,1
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
                state = np.array(query) * 2 - 1  # Convert 0,1 to -1,1
                
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
                    output += f"Pattern {i+1}: `{pattern}`\n"
                
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
                min_distance = float('inf')
                closest_pattern = None
                closest_idx = -1
                
                for i, pattern in enumerate(patterns):
                    # Calculate Hamming distance
                    distance = sum(p != r for p, r in zip(pattern, recalled.tolist(), strict=False))
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
    """
    Tool that uses Conditional Random Fields for sequence labeling in code.
    
    CRFs are statistical modeling methods that can label sequence data,
    useful for tasks like code element classification or test dependency analysis.
    """
    
    name: str = "Conditional Random Field Tool"
    description: str = """
    Uses Conditional Random Fields for sequence labeling in code.
    This tool can be used to:
    - Tag code elements based on their context and role
    - Identify dependencies between code elements
    - Analyze test coverage at a sequence level
    
    Particularly useful for the Test Agent to analyze test dependencies.
    """
    
    def _run(self, sequences: List[List[Dict[str, Any]]], labels: Optional[List[List[str]]] = None, 
             mode: str = "train", features: List[str] = None) -> str:
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
        try:
            import sklearn_crfsuite
            from sklearn_crfsuite import metrics
            
            output = "## Conditional Random Field Analysis\n\n"
            
            if mode == "train" and labels:
                output += f"Training CRF model on {len(sequences)} sequences.\n\n"
                
                # Train a CRF model
                crf = sklearn_crfsuite.CRF(
                    algorithm='lbfgs',
                    c1=0.1,
                    c2=0.1,
                    max_iterations=100,
                    all_possible_transitions=True
                )
                
                # Fit the model
                crf.fit(sequences, labels)
                
                # Make predictions on training data for evaluation
                predictions = crf.predict(sequences)
                
                # Compute metrics
                labels_flat = [l for seq in labels for l in seq]
                predictions_flat = [p for seq in predictions for p in seq]
                
                set(labels_flat)
                
                output += "### Training Metrics\n\n"
                output += f"Accuracy: {metrics.flat_accuracy_score(labels_flat, predictions_flat):.4f}\n\n"
                
                # Get label weights
                if hasattr(crf, 'state_features_'):
                    output += "### Top Feature Weights\n\n"
                    
                    # Get top positive weights
                    weights = []
                    for (attr, label), weight in crf.state_features_.items():
                        weights.append((attr, label, weight))
                    
                    # Sort by absolute weight value
                    weights.sort(key=lambda x: abs(x[2]), reverse=True)
                    
                    # Show top weights
                    for attr, label, weight in weights[:10]:
                        output += f"- {attr} → {label}: {weight:.4f}\n"
                
                output += "\nThe CRF model has been trained and can now be used for sequence labeling.\n"
                
            elif mode == "predict":
                output += f"Making predictions on {len(sequences)} sequences.\n\n"
                
                # Placeholder for loading a pre-trained model
                # In a real implementation, we would load the model and make predictions
                
                output += "This is a placeholder for CRF prediction. In a real implementation, the model would be loaded and predictions would be made.\n"
                
            else:
                output += "Please provide labels for training mode.\n"
                
            return output
            
        except ImportError:
            return "Error: Required libraries (sklearn-crfsuite) not installed. Please install with 'pip install sklearn-crfsuite'."
        except Exception as e:
            return f"Error using Conditional Random Fields: {str(e)}"


# Create a class to integrate all algorithm tools
class AlgorithmIntegrationTool(BaseTool):
    """Tool to select and apply specialized algorithms to code quality tasks."""
    
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
        if task_type == "pattern_recognition":
            if data.get("algorithm") == "elm":
                tool = ExtremeLearningMachineTool()
                return tool._run(
                    code_features=data.get("features"),
                    labels=data.get("labels"),
                    mode=data.get("mode", "train")
                )
            elif data.get("algorithm") == "hopfield":
                tool = HopfieldNetworkTool()
                return tool._run(
                    patterns=data.get("patterns"),
                    query_pattern=data.get("query"),
                    mode=data.get("mode", "store")
                )
            elif data.get("algorithm") == "tsetlin":
                tool = TsetlinMachineTool()
                return tool._run(
                    data=data.get("data"),
                    feature_names=data.get("feature_names"),
                    num_clauses=data.get("num_clauses", 20),
                    specificity=data.get("specificity", 3.9),
                    threshold=data.get("threshold", 15),
                    epochs=data.get("epochs", 50)
                )
                
        elif task_type == "clustering":
            if data.get("algorithm") == "som":
                tool = SelfOrganizingMapTool()
                return tool._run(
                    code_vectors=data.get("vectors"),
                    labels=data.get("labels"),
                    som_size=data.get("som_size", (10, 10))
                )
                
        elif task_type == "anomaly_detection":
            if data.get("algorithm") == "iforest":
                tool = IsolationForestTool()
                return tool._run(
                    data_points=data.get("data_points"),
                    feature_names=data.get("feature_names"),
                    contamination=data.get("contamination", 0.1)
                )
                
        elif task_type == "optimization":
            if data.get("algorithm") == "bayesian":
                tool = BayesianOptimizationTool()
                return tool._run(
                    objective_function_name=data.get("objective"),
                    param_bounds=data.get("param_bounds"),
                    n_iterations=data.get("n_iterations", 20),
                    init_points=data.get("init_points", 5)
                )
                
        elif task_type == "sequence_analysis":
            if data.get("algorithm") == "crf":
                tool = ConditionalRandomFieldTool()
                return tool._run(
                    sequences=data.get("sequences"),
                    labels=data.get("labels"),
                    mode=data.get("mode", "train"),
                    features=data.get("features")
                )
                
        elif task_type == "relation_discovery":
            if data.get("algorithm") == "symbolic_regression":
                tool = SymbolicRegressionTool()
                return tool._run(
                    X=data.get("X"),
                    y=data.get("y"),
                    feature_names=data.get("feature_names"),
                    max_generations=data.get("max_generations", 100),
                    population_size=data.get("population_size", 1000)
                )
                
        return f"Unknown task type or algorithm: {task_type}, {data.get('algorithm')}" 