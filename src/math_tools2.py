"""
Advanced mathematical tools for code representation and algorithm quality assessment.
"""

import ast
import math
import os
from typing import Any, Dict, List, Optional

import chromadb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from crewai.tools import BaseTool
from sklearn.decomposition import PCA

# Initialize ChromaDB client for vector storage
chroma_client = chromadb.Client()
try:
    code_representation_collection = chroma_client.get_or_create_collection(
        name="code_representations",
        metadata={"hnsw:space": "cosine"}
    )
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")


class CodeMathRepresentationTool(BaseTool):
    """Tool for converting code into mathematical representations."""
    
    name: str = "Code Math Representation Tool"
    description: str = """
    Converts Python code into mathematical representations using various metrics:
    - Control flow complexity (cyclomatic complexity)
    - Information density (Halstead metrics)
    - Dependency graphs and their spectral properties
    - Algorithm efficiency metrics (time/space complexity)
    """

    def _run(self, code_str: str, calculate_spectra: bool = True) -> Dict[str, Any]:
        """
        Convert code to mathematical representation.
        
        Args:
            code_str: String containing Python code
            calculate_spectra: Whether to calculate spectral properties
            
        Returns:
            Dictionary with mathematical metrics
        """
        try:
            # Parse the code
            tree = ast.parse(code_str)
            
            # Extract basic metrics
            metrics = self._extract_basic_metrics(tree)
            
            # Build and analyze control flow graph
            cfg = self._build_control_flow_graph(tree)
            cfg_metrics = self._analyze_graph(cfg, 'Control Flow', calculate_spectra)
            metrics.update(cfg_metrics)
            
            # Build and analyze data dependency graph
            ddg = self._build_data_dependency_graph(tree)
            ddg_metrics = self._analyze_graph(ddg, 'Data Dependency', calculate_spectra)
            metrics.update(ddg_metrics)
            
            # Add algorithmic complexity estimates
            metrics.update(self._estimate_algorithmic_complexity(tree))
            
            return metrics
            
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_basic_metrics(self, tree: ast.AST) -> Dict[str, float]:
        """Extract basic code metrics."""
        # Count various node types
        node_counts = {}
        for node_type in ast.AST.__subclasses__():
            type_name = node_type.__name__
            node_counts[type_name] = len([n for n in ast.walk(tree) if isinstance(n, node_type)])
        
        # Count total nodes, lines
        total_nodes = sum(node_counts.values())
        lines = len(ast.unparse(tree).split('\n'))
        
        # Calculate Halstead metrics
        operators = len([n for n in ast.walk(tree) if isinstance(n, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare))])
        operands = len([n for n in ast.walk(tree) if isinstance(n, (ast.Name, ast.Num, ast.Str))])
        
        # Unique operators and operands (approximation)
    `        unique_operators = len(set([type(n).__name__ for n in ast.walk(tree) if isinstance(n, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare))]))
`        unique_operands = len(set([getattr(n, 'id', str(n)) for n in ast.walk(tree) if isinstance(n, (ast.Name, ast.Num, ast.Str))]))
        
        # Halstead volume: N * log2(n) where N = operators + operands, n = unique_operators + unique_operands
        n = max(1, unique_operators + unique_operands)  # Prevent log(0)
        N = operators + operands
        volume = N * math.log2(n) if N > 0 and n > 1 else 0
        
        # Information density
        information_density = volume / total_nodes if total_nodes > 0 else 0
        
        return {
            "total_nodes": total_nodes,
            "lines_of_code": lines,
            "operators": operators,
            "operands": operands,
            "unique_operators": unique_operators,
            "unique_operands": unique_operands,
            "halstead_volume": volume,
            "information_density": information_density,
        }
    
    def _build_control_flow_graph(self, tree: ast.AST) -> nx.DiGraph:
        """Build a control flow graph from AST."""
        G = nx.DiGraph()
        
        # Create nodes for each statement
        for _i, node in enumerate(ast.walk(tree)):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.Try)):
                node_id = f"{type(node).__name__}_{id(node)}"
                G.add_node(node_id, type=type(node).__name__)
                
                # Add edges for control flow
                if isinstance(node, ast.If):
                    # Add true and false branches
                    true_id = f"IfBody_{id(node)}"
                    false_id = f"ElseBody_{id(node)}"
                    G.add_node(true_id, type="Body")
                    G.add_edge(node_id, true_id, type="True")
                    
                    if getattr(node, 'orelse', None):
                        G.add_node(false_id, type="Body")
                        G.add_edge(node_id, false_id, type="False")
                
                elif isinstance(node, (ast.For, ast.While)):
                    body_id = f"LoopBody_{id(node)}"
                    G.add_node(body_id, type="Body")
                    G.add_edge(node_id, body_id, type="Body")
                    G.add_edge(body_id, node_id, type="Loop")  # Loop back
                
                elif isinstance(node, ast.Try):
                    try_id = f"TryBody_{id(node)}"
                    G.add_node(try_id, type="Body")
                    G.add_edge(node_id, try_id, type="Try")
                    
                    for handler in getattr(node, 'handlers', []):
                        except_id = f"ExceptBody_{id(handler)}"
                        G.add_node(except_id, type="Body")
                        G.add_edge(node_id, except_id, type="Except")
        
        # Ensure graph is not empty
        if not G.nodes:
            G.add_node("empty")
            
        return G
    
    def _build_data_dependency_graph(self, tree: ast.AST) -> nx.DiGraph:
        """Build a data dependency graph from AST."""
        G = nx.DiGraph()
        
        # Track variable assignments and usage
        var_assignments = {}
        
        for node in ast.walk(tree):
            # Variable assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_id = f"Var_{target.id}"
                        assign_id = f"Assign_{id(node)}"
                        
                        G.add_node(var_id, type="Variable", name=target.id)
                        G.add_node(assign_id, type="Assignment")
                        G.add_edge(assign_id, var_id, type="Defines")
                        
                        var_assignments[target.id] = assign_id
                        
                        # Add edges from variables used in the assignment
                        for name_node in [n for n in ast.walk(node.value) if isinstance(n, ast.Name)]:
                            if name_node.id in var_assignments:
                                used_var_id = f"Var_{name_node.id}"
                                G.add_node(used_var_id, type="Variable", name=name_node.id)
                                G.add_edge(used_var_id, assign_id, type="UsedIn")
            
            # Variable usage in expressions
            elif isinstance(node, ast.Expr) and hasattr(node, 'value'):
                expr_id = f"Expr_{id(node)}"
                G.add_node(expr_id, type="Expression")
                
                for name_node in [n for n in ast.walk(node.value) if isinstance(n, ast.Name)]:
                    if name_node.id in var_assignments:
                        var_id = f"Var_{name_node.id}"
                        G.add_node(var_id, type="Variable", name=name_node.id)
                        G.add_edge(var_id, expr_id, type="UsedIn")
        
        # Ensure graph is not empty
        if not G.nodes:
            G.add_node("empty")
            
        return G
    
    def _analyze_graph(self, G: nx.DiGraph, prefix: str, calculate_spectra: bool) -> Dict[str, float]:
        """Analyze a graph and extract its properties."""
        metrics = {}
        
        # Basic graph metrics
        metrics[f"{prefix}_nodes"] = G.number_of_nodes()
        metrics[f"{prefix}_edges"] = G.number_of_edges()
        metrics[f"{prefix}_density"] = nx.density(G)
        
        # Connectivity
        if G.number_of_nodes() > 1:
            try:
                metrics[f"{prefix}_avg_shortest_path"] = nx.average_shortest_path_length(G)
            except (nx.NetworkXError, nx.NetworkXNoPath):
                # Graph may not be connected
                metrics[f"{prefix}_avg_shortest_path"] = 0
        else:
            metrics[f"{prefix}_avg_shortest_path"] = 0
            
        # Centrality measures
        if G.number_of_nodes() > 1:
            try:
                centrality = nx.degree_centrality(G)
                metrics[f"{prefix}_max_centrality"] = max(centrality.values()) if centrality else 0
                metrics[f"{prefix}_avg_centrality"] = sum(centrality.values()) / len(centrality) if centrality else 0
            except Exception:
                metrics[f"{prefix}_max_centrality"] = 0
                metrics[f"{prefix}_avg_centrality"] = 0
        else:
            metrics[f"{prefix}_max_centrality"] = 0
            metrics[f"{prefix}_avg_centrality"] = 0
            
        # Cyclomatic complexity approximation
        metrics[f"{prefix}_cyclomatic"] = G.number_of_edges() - G.number_of_nodes() + 2
        
        # Spectral properties (eigenvalues of adjacency matrix)
        if calculate_spectra and G.number_of_nodes() > 1:
            try:
                # Use only the largest eigenvalues for efficiency
                eigenvalues = sorted(nx.linalg.adj_matrix(G).todense().real, reverse=True)[:3]
                for i, val in enumerate(eigenvalues):
                    metrics[f"{prefix}_eigenvalue_{i}"] = float(val)
            except Exception:
                metrics[f"{prefix}_eigenvalue_0"] = 0
                metrics[f"{prefix}_eigenvalue_1"] = 0
                metrics[f"{prefix}_eigenvalue_2"] = 0
        else:
            metrics[f"{prefix}_eigenvalue_0"] = 0
            metrics[f"{prefix}_eigenvalue_1"] = 0
            metrics[f"{prefix}_eigenvalue_2"] = 0
                
        return metrics
    
    def _estimate_algorithmic_complexity(self, tree: ast.AST) -> Dict[str, float]:
        """Estimate algorithmic complexity."""
        metrics = {}
        
        # Count nested loops as an indicator of time complexity
        loop_depths = []
        current_depth = 0
        
        class LoopVisitor(ast.NodeVisitor):
            def visit_For(self, node):
                nonlocal current_depth
                current_depth += 1
                loop_depths.append(current_depth)
                self.generic_visit(node)
                current_depth -= 1
                
            def visit_While(self, node):
                nonlocal current_depth
                current_depth += 1
                loop_depths.append(current_depth)
                self.generic_visit(node)
                current_depth -= 1
        
        LoopVisitor().visit(tree)
        
        # Determine time complexity class
        max_depth = max(loop_depths) if loop_depths else 0
        
        # Complexity classes: O(1), O(log n), O(n), O(n log n), O(n²), O(n³), O(2^n)
        if max_depth == 0:
            metrics["time_complexity_class"] = 1  # O(1)
        elif max_depth == 1:
            metrics["time_complexity_class"] = 2  # O(n)
        elif max_depth == 2:
            metrics["time_complexity_class"] = 4  # O(n²)
        elif max_depth == 3:
            metrics["time_complexity_class"] = 5  # O(n³)
        else:
            metrics["time_complexity_class"] = 6  # O(2^n) or worse
            
        # Count variable assignments as indicator of space complexity
        var_assignments = len([n for n in ast.walk(tree) if isinstance(n, ast.Assign)])
        
        if var_assignments <= 5:
            metrics["space_complexity_class"] = 1  # O(1)
        elif var_assignments <= 20:
            metrics["space_complexity_class"] = 2  # O(n)
        else:
            metrics["space_complexity_class"] = 3  # O(n²) or worse
            
        metrics["max_loop_depth"] = max_depth
        metrics["variable_assignments"] = var_assignments
        
        return metrics


class CodeVectorizationTool(BaseTool):
    """Tool for converting code mathematical representations to vector embeddings."""
    
    name: str = "Code Vectorization Tool"
    description: str = """
    Converts mathematical code representations into vector embeddings:
    - Normalizes and scales metrics
    - Computes a compact vector representation
    - Stores vectors in a vector database for similarity search
    """

    def _run(self, code_id: str, math_representation: Dict[str, Any], 
             store_in_db: bool = True) -> List[float]:
        """
        Convert mathematical representation to vector embedding.
        
        Args:
            code_id: Unique identifier for this code snippet
            math_representation: Dictionary of mathematical metrics
            store_in_db: Whether to store result in vector database
            
        Returns:
            Vector embedding as a list of floats
        """
        try:
            # Remove non-numeric and error fields
            filtered_metrics = {k: v for k, v in math_representation.items() 
                              if isinstance(v, (int, float)) and k != "error"}
            
            # Sort keys for consistent ordering
            sorted_keys = sorted(filtered_metrics.keys())
            
            # Create vector from values
            vector = [filtered_metrics[k] for k in sorted_keys]
            
            # Normalize vector
            norm = np.linalg.norm(vector)
            normalized_vector = [v / norm if norm > 0 else 0 for v in vector]
            
            # Store in vector DB if requested
            if store_in_db:
                try:
                    metadata = {
                        "keys": ",".join(sorted_keys),
                        "source": code_id
                    }
                    code_representation_collection.upsert(
                        ids=[code_id],
                        embeddings=[normalized_vector],
                        metadatas=[metadata]
                    )
                except Exception as e:
                    print(f"Error storing in vector DB: {e}")
            
            return normalized_vector
            
        except Exception as e:
            print(f"Error in vectorization: {e}")
            return [0.0] * 10  # Return zero vector as fallback


class AlgorithmQualityAssessmentTool(BaseTool):
    """Tool for assessing algorithm quality in 3D space."""
    
    name: str = "Algorithm Quality Assessment Tool"
    description: str = """
    Assesses algorithm quality by placing code representations in 3D space:
    - Projects high-dimensional vectors to 3D
    - Visualizes algorithms in relation to known good/bad examples
    - Provides quality scores based on proximity to exemplars
    """

    def _run(self, code_id: str, vector: List[float], 
             reference_patterns: Optional[Dict[str, List[float]]] = None,
             visualize: bool = True) -> Dict[str, Any]:
        """
        Assess algorithm quality by visualizing in 3D space.
        
        Args:
            code_id: Identifier for this code snippet
            vector: Vector embedding of code
            reference_patterns: Known good/bad algorithm patterns
            visualize: Whether to generate 3D visualization
            
        Returns:
            Dictionary with quality assessment results
        """
        try:
            # Make sure vector is a properly sized numpy array
            vector_np = np.array(vector)
            
            # Default reference patterns if none provided
            if not reference_patterns:
                # Create default patterns with matching dimensions to input vector
                vec_length = len(vector)
                optimal_time = [0.1] * vec_length
                optimal_space = [0.2] * vec_length
                poor_quality = [0.8] * vec_length
                
                reference_patterns = {
                    "optimal_time": optimal_time,
                    "optimal_space": optimal_space,
                    "poor_quality": poor_quality
                }
            
            # Ensure all reference patterns have the same dimensions
            ref_patterns_normalized = {}
            for name, pattern in reference_patterns.items():
                if len(pattern) != len(vector):
                    # Adjust pattern to match vector length
                    if len(pattern) > len(vector):
                        ref_patterns_normalized[name] = pattern[:len(vector)]
                    else:
                        extended = pattern + [0.0] * (len(vector) - len(pattern))
                        ref_patterns_normalized[name] = extended
                else:
                    ref_patterns_normalized[name] = pattern
            
            # Add vectorized code to reference patterns for visualization
            all_vectors = ref_patterns_normalized.copy()
            all_vectors[code_id] = vector
            
            # Skip vector DB query, we'll implement directly
            similar_codes = []
            
            # Project vectors to 3D using PCA
            # Convert all vectors to a 2D numpy array with consistent shape
            vectors_list = []
            pattern_names = []
            for name, vec in all_vectors.items():
                vectors_list.append(np.array(vec, dtype=float))
                pattern_names.append(name)
                
            # Stack vectors into a 2D array
            all_vectors_array = np.vstack(vectors_list)
            
            if all_vectors_array.size > 0:
                # Use PCA to reduce dimensions to 3
                n_components = min(3, all_vectors_array.shape[1], all_vectors_array.shape[0])
                pca = PCA(n_components=n_components)
                vectors_3d = pca.fit_transform(all_vectors_array)
                
                # Create mapping of names to 3D coordinates
                coords_3d = {name: vectors_3d[i].tolist() for i, name in enumerate(pattern_names)}
                
                # Calculate quality scores based on distances to reference patterns
                distances = {}
                for pattern_name, pattern_vector in ref_patterns_normalized.items():
                    pattern_np = np.array(pattern_vector, dtype=float)
                    distance = np.linalg.norm(vector_np - pattern_np)
                    distances[pattern_name] = float(distance)
                
                # Quality score: closer to optimal patterns, further from poor patterns
                quality_score = 0.0
                if "optimal_time" in distances and "optimal_space" in distances and "poor_quality" in distances:
                    # Higher score means better quality (0-100 scale)
                    optimal_dist = min(distances["optimal_time"], distances["optimal_space"])
                    poor_dist = distances["poor_quality"]
                    
                    # Score formula: 100 * (poor_dist / (poor_dist + optimal_dist))
                    if poor_dist + optimal_dist > 0:
                        quality_score = 100 * (poor_dist / (poor_dist + optimal_dist))
                        quality_score = min(100, max(0, quality_score))  # Clamp to 0-100
                
                # Generate visualization if requested
                if visualize:
                    self._generate_3d_visualization(coords_3d, code_id)
                
                return {
                    "quality_score": quality_score,
                    "coordinates_3d": coords_3d[code_id],
                    "distances": distances,
                    "similar_algorithms": similar_codes,
                    "explained_variance": pca.explained_variance_ratio_.tolist() if hasattr(pca, 'explained_variance_ratio_') else [],
                }
            else:
                return {"error": "No vectors to process"}
            
        except Exception as e:
            print(f"Error in algorithm quality assessment: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _generate_3d_visualization(self, coords_3d: Dict[str, List[float]], highlight_id: str) -> None:
        """Generate 3D visualization of algorithm vectors."""
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Color mapping for different types of points
            colors = {
                "optimal_time": "blue",
                "optimal_space": "green",
                "poor_quality": "red",
                highlight_id: "yellow"
            }
            
            # Plot each point
            for name, coords in coords_3d.items():
                color = colors.get(name, "gray")
                marker = 'o' if name != highlight_id else '*'
                size = 100 if name != highlight_id else 200
                ax.scatter(coords[0], coords[1], coords[2], color=color, s=size, marker=marker, label=name)
            
            # Add labels and legend
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
            ax.set_title('Algorithm Quality Assessment in 3D Space')
            ax.legend()
            
            # Save the figure with absolute path
            current_dir = os.path.abspath(os.getcwd())
            output_path = os.path.join(current_dir, "algorithm_quality_3d.png")
            plt.savefig(output_path)
            plt.close()
            print(f"3D visualization saved to: {output_path}")
        except Exception as e:
            print(f"Error generating 3D visualization: {e}")


class APIAnalysisTool(BaseTool):
    """Tool for analyzing API quality using mathematical metrics."""
    
    name: str = "API Analysis Tool"
    description: str = """
    Analyzes API quality using mathematical metrics:
    - API design cohesion and coupling
    - Interface complexity and consistency
    - Parameter optimization
    """

    def _run(self, api_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze API quality using mathematical metrics.
        
        Args:
            api_spec: API specification including endpoints, parameters, etc.
            
        Returns:
            Analysis results with metrics and recommendations
        """
        try:
            # Extract endpoints, parameters, response types
            endpoints = api_spec.get("endpoints", [])
            if not endpoints:
                return {"error": "No endpoints found in API spec"}
                
            # Calculate basic metrics
            num_endpoints = len(endpoints)
            total_params = sum(len(endpoint.get("parameters", [])) for endpoint in endpoints)
            avg_params = total_params / num_endpoints if num_endpoints > 0 else 0
            
            # Analyze parameter consistency
            param_usage = {}
            for endpoint in endpoints:
                for param in endpoint.get("parameters", []):
                    param_name = param.get("name", "")
                    if param_name:
                        if param_name not in param_usage:
                            param_usage[param_name] = {
                                "count": 0,
                                "types": set(),
                                "endpoints": []
                            }
                        param_usage[param_name]["count"] += 1
                        param_usage[param_name]["types"].add(param.get("type", "unknown"))
                        param_usage[param_name]["endpoints"].append(endpoint.get("name", "unnamed"))
            
            # Calculate consistency metrics
            consistent_params = [p for p, data in param_usage.items() 
                               if len(data["types"]) == 1 and data["count"] > 1]
            inconsistent_params = [p for p, data in param_usage.items() 
                                 if len(data["types"]) > 1]
            
            # Build endpoint similarity matrix for cohesion analysis
            endpoint_matrix = np.zeros((num_endpoints, num_endpoints))
            for i, e1 in enumerate(endpoints):
                e1_params = {p.get("name") for p in e1.get("parameters", [])}
                for j, e2 in enumerate(endpoints):
                    if i != j:
                        e2_params = {p.get("name") for p in e2.get("parameters", [])}
                        # Jaccard similarity of parameters
                        intersection = len(e1_params.intersection(e2_params))
                        union = len(e1_params.union(e2_params))
                        similarity = intersection / union if union > 0 else 0
                        endpoint_matrix[i, j] = similarity
            
            # Calculate cohesion metrics
            avg_similarity = np.mean(endpoint_matrix)
            max_similarity = np.max(endpoint_matrix) if endpoint_matrix.size > 0 else 0
            
            # Create graph representation for coupling analysis
            G = nx.DiGraph()
            
            # Add nodes for endpoints
            for i, endpoint in enumerate(endpoints):
                endpoint_name = endpoint.get("name", f"endpoint_{i}")
                G.add_node(endpoint_name, type="endpoint")
            
            # Add edges for dependencies (shared resources, parameters)
            for i, e1 in enumerate(endpoints):
                e1_name = e1.get("name", f"endpoint_{i}")
                e1_resources = set(e1.get("resources", []))
                
                for j, e2 in enumerate(endpoints):
                    if i != j:
                        e2_name = e2.get("name", f"endpoint_{j}")
                        e2_resources = set(e2.get("resources", []))
                        
                        # Add edge if resources overlap
                        shared_resources = e1_resources.intersection(e2_resources)
                        if shared_resources:
                            G.add_edge(e1_name, e2_name, weight=len(shared_resources))
            
            # Calculate coupling metrics
            coupling_degree = nx.degree_centrality(G) if G.number_of_nodes() > 0 else {}
            avg_coupling = sum(coupling_degree.values()) / len(coupling_degree) if coupling_degree else 0
            max_coupling = max(coupling_degree.values()) if coupling_degree else 0
            
            # Calculate overall API quality score (0-100)
            # Higher consistency, lower coupling, balanced parameters → better score
            param_balance_score = 100 * (1.0 / (1.0 + abs(avg_params - 3.0)))  # Ideal: ~3 params per endpoint
            consistency_score = 100 * (len(consistent_params) / total_params if total_params > 0 else 0)
            coupling_score = 100 * (1.0 - avg_coupling)  # Lower coupling is better
            
            # Overall quality score (weighted average)
            quality_score = (0.3 * param_balance_score + 
                            0.4 * consistency_score + 
                            0.3 * coupling_score)
            
            # Generate recommendations
            recommendations = []
            
            if inconsistent_params:
                recommendations.append(
                    f"Standardize parameter types for: {', '.join(inconsistent_params[:3])}" +
                    (f" and {len(inconsistent_params) - 3} more" if len(inconsistent_params) > 3 else "")
                )
                
            if avg_params > 5:
                recommendations.append(
                    "Consider reducing parameter count for endpoints - use request objects for complex operations"
                )
                
            if avg_coupling > 0.6:
                recommendations.append(
                    "High API coupling detected - consider reorganizing endpoint dependencies"
                )
                
            return {
                "metrics": {
                    "num_endpoints": num_endpoints,
                    "total_parameters": total_params,
                    "avg_parameters_per_endpoint": avg_params,
                    "consistent_parameters": len(consistent_params),
                    "inconsistent_parameters": len(inconsistent_params),
                    "avg_endpoint_similarity": float(avg_similarity),
                    "max_endpoint_similarity": float(max_similarity),
                    "avg_coupling": avg_coupling,
                    "max_coupling": max_coupling
                },
                "scores": {
                    "parameter_balance": param_balance_score,
                    "consistency": consistency_score,
                    "coupling": coupling_score,
                    "overall_quality": quality_score
                },
                "recommendations": recommendations
            }
                
        except Exception as e:
            return {"error": str(e)}


def code_to_quality_assessment(code_str: str) -> Dict[str, Any]:
    """
    End-to-end function to convert code to quality assessment.
    
    Args:
        code_str: Python code as string
        
    Returns:
        Quality assessment results
    """
    # Create unique ID for this code
    import hashlib
    code_id = hashlib.md5(code_str.encode()).hexdigest()
    
    # Initialize tools
    math_tool = CodeMathRepresentationTool()
    vector_tool = CodeVectorizationTool()
    quality_tool = AlgorithmQualityAssessmentTool()
    
    # Process the code
    math_repr = math_tool._run(code_str)
    vector = vector_tool._run(code_id, math_repr, store_in_db=False)
    assessment = quality_tool._run(code_id, vector, visualize=True)
    
    return {
        "code_id": code_id,
        "mathematical_representation": math_repr,
        "quality_assessment": assessment,
    } 