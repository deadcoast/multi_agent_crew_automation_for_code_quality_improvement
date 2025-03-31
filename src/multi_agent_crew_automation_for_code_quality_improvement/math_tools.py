"""
Mathematical tools for enhancing code quality analysis and improvement.
These tools implement various mathematical algorithms and techniques 
as described in the Coding Crew Design documents.
"""

import ast
import glob
import os

import networkx as nx
import numpy as np
import radon.complexity as radon_cc
import radon.metrics as radon_metrics
import radon.raw as radon_raw
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from crewai.tools import BaseTool
except ImportError:
    # Fallback implementation if crewai_tools is not available
    class BaseTool:
        """Base class for all tools."""
        name: str = "BaseTool"
        description: str = "Base class for all tools."
        
        def _run(self, *args, **kwargs) -> str:
            raise NotImplementedError("Subclasses must implement this method")
        
        def run(self, *args, **kwargs) -> str:
            return self._run(*args, **kwargs)


class CodeComplexityAnalysisTool(BaseTool):
    """Tool for analyzing code complexity using radon metrics."""
    
    name: str = "Code Complexity Analysis Tool"
    description: str = """
    Analyzes Python code complexity using metrics such as:
    - Cyclomatic Complexity
    - Halstead Volume
    - Maintainability Index
    
    Provides a ranked list of files or functions based on these metrics
    to help prioritize refactoring efforts.
    """

    def _run(self, file_patterns: str, limit: int = 10) -> str:
        """
        Analyze code complexity for the given file patterns.
        
        Args:
            file_patterns: Glob patterns for Python files to analyze (comma-separated)
            limit: Maximum number of files to show in the results
            
        Returns:
            A summary of complexity metrics for the most complex files
        """
        results = []
        patterns = [p.strip() for p in file_patterns.split(',')]
        
        for pattern in patterns:
            for file_path in glob.glob(pattern, recursive=True):
                if not file_path.endswith('.py'):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                        
                    # Calculate complexity metrics
                    cc_results = radon_cc.cc_visit(code)
                    mi = radon_metrics.mi_visit(code, multi=True)
                    raw_metrics = radon_raw.analyze(code)
                    
                    # Calculate average complexity for the file
                    avg_complexity = np.mean([r.complexity for r in cc_results]) if cc_results else 0
                    
                    # Append results
                    results.append({
                        'file': file_path,
                        'avg_complexity': avg_complexity,
                        'maintainability_index': mi,
                        'loc': raw_metrics.loc,
                        'functions': len(cc_results),
                        'comments_ratio': raw_metrics.comments / raw_metrics.loc if raw_metrics.loc > 0 else 0,
                        'complexity_details': {r.name: r.complexity for r in cc_results}
                    })
                except Exception as e:
                    results.append({
                        'file': file_path,
                        'error': str(e)
                    })
        
        # Sort results by average complexity
        sorted_results = sorted(results, key=lambda x: x.get('avg_complexity', 0), reverse=True)
        limited_results = sorted_results[:limit]
        
        # Format the results
        output = "## Code Complexity Analysis Results\n\n"
        output += f"Analyzed {len(results)} Python files. Top {len(limited_results)} most complex files:\n\n"
        
        for i, r in enumerate(limited_results, 1):
            output += f"### {i}. {r['file']}\n"
            if 'error' in r:
                output += f"Error: {r['error']}\n"
                continue
                
            output += f"- Average Cyclomatic Complexity: {r['avg_complexity']:.2f}\n"
            output += f"- Maintainability Index: {r['maintainability_index']:.2f}\n"
            output += f"- Lines of Code: {r['loc']}\n"
            output += f"- Number of Functions: {r['functions']}\n"
            output += f"- Comments Ratio: {r['comments_ratio']:.2f}\n"
            
            if r['complexity_details']:
                output += "- Complex Functions:\n"
                complex_funcs = sorted(r['complexity_details'].items(), key=lambda x: x[1], reverse=True)[:3]
                for name, complexity in complex_funcs:
                    output += f"  - `{name}`: {complexity}\n"
            
            output += "\n"
        
        return output

class DependencyGraphAnalysisTool(BaseTool):
    """Tool for analyzing module dependencies using networkx."""
    
    name: str = "Dependency Graph Analysis Tool"
    description: str = """
    Analyzes Python module dependencies by:
    - Building a directed graph of import relationships
    - Identifying highly connected modules
    - Detecting circular dependencies
    - Suggesting potential refactoring opportunities
    """

    def _run(self, directory: str) -> str:
        """
        Analyze dependencies between Python modules in the given directory.
        
        Args:
            directory: Directory containing Python files to analyze
            
        Returns:
            A summary of dependency analysis results
        """
        # Build dependency graph
        G = nx.DiGraph()

        # Track module imports
        imports_by_file = {}

        # Process all Python files
        for root, _, files in os.walk(directory):
            for file in files:
                if not file.endswith('.py'):
                    continue

                file_path = os.path.join(root, file)
                module_name = os.path.relpath(file_path, directory).replace('/', '.').replace('\\', '.')[:-3]
                G.add_node(module_name)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()

                    # Parse imports using AST
                    tree = ast.parse(code)
                    imports = []

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            imports.extend(name.name for name in node.names)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.append(node.module)

                    imports_by_file[module_name] = imports

                    # Add edges to the graph
                    for imp in imports:
                        if imp in G:
                            G.add_edge(module_name, imp)

                except Exception:
                    continue

        output = "## Dependency Graph Analysis Results\n\n" + "### Graph Statistics\n"
        output += f"- Total modules: {G.number_of_nodes()}\n"
        output += f"- Total dependencies: {G.number_of_edges()}\n"

        # 2. Identify highly connected modules (hubs)
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())

        top_imported = sorted(list(in_degree.items()), key=lambda x: x[1], reverse=True)[:5]
        top_importers = sorted(list(out_degree.items()), key=lambda x: x[1], reverse=True)[:5]

        output += "\n### Most Imported Modules\n"
        for module, count in top_imported:
            output += f"- `{module}`: imported by {count} modules\n"

        output += "\n### Modules with Most Dependencies\n"
        for module, count in top_importers:
            output += f"- `{module}`: imports {count} modules\n"

        # 3. Detect circular dependencies
        try:
            cycles = list(nx.simple_cycles(G))
            output += "\n### Circular Dependencies\n"

            if cycles:
                output += f"Found {len(cycles)} circular dependencies:\n"
                for i, cycle in enumerate(cycles[:5], 1):
                    output += f"{i}. {' → '.join(cycle)} → {cycle[0]}\n"

                if len(cycles) > 5:
                    output += f"... and {len(cycles) - 5} more\n"
            else:
                output += "No circular dependencies detected.\n"

        except Exception as e:
            output += f"\nError detecting circular dependencies: {str(e)}\n"

        # 4. Community detection
        try:
            communities = nx.community.greedy_modularity_communities(G.to_undirected())
            output += "\n### Module Communities\n"
            output += f"Detected {len(communities)} module communities:\n\n"

            for i, community in enumerate(list(communities)[:3], 1):
                output += f"Community {i} ({len(community)} modules):\n"
                for module in sorted(community)[:5]:
                    output += f"- {module}\n"
                if len(community) > 5:
                    output += f"... and {len(community) - 5} more\n"
                output += "\n"

            if len(communities) > 3:
                output += f"... and {len(communities) - 3} more communities\n"

        except Exception as e:
            output += f"\nError detecting communities: {str(e)}\n"

        return output

class CodeClusteringTool(BaseTool):
    """Tool for clustering similar code files using machine learning."""
    
    name: str = "Code Clustering Tool"
    description: str = """
    Clusters Python files based on their similarity using ML techniques:
    - TF-IDF vectorization of code content
    - K-means or DBSCAN clustering
    - Visualization of clusters
    
    Useful for identifying groups of related files for batch refactoring.
    """

    def _run(self, file_patterns: str, n_clusters: int = 5, algorithm: str = "kmeans") -> str:
        """
        Cluster similar Python files using machine learning.
        
        Args:
            file_patterns: Glob patterns for Python files to analyze (comma-separated)
            n_clusters: Number of clusters to create (for K-means)
            algorithm: Clustering algorithm to use ("kmeans" or "dbscan")
            
        Returns:
            A summary of clustering results and file similarities
        """
        # Collect files
        file_contents = {}
        patterns = [p.strip() for p in file_patterns.split(',')]
        
        for pattern in patterns:
            for file_path in glob.glob(pattern, recursive=True):
                if not file_path.endswith('.py'):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_contents[file_path] = f.read()
                except Exception:
                    continue
        
        if len(file_contents) < 2:
            return "Not enough files found for clustering. Need at least 2 files."
        
        # Convert files to TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000,
                                     stop_words='english',
                                     token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]*\b')
        
        file_paths = list(file_contents.keys())
        content_list = list(file_contents.values())
        
        try:
            vectors = vectorizer.fit_transform(content_list)
            
            # Perform clustering
            clusters = {}
            cluster_method = ""
            
            if algorithm.lower() == "kmeans":
                n_clusters = min(n_clusters, len(file_contents) - 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(vectors)
                cluster_method = f"K-means (k={n_clusters})"
                
            elif algorithm.lower() == "dbscan":
                dbscan = DBSCAN(eps=0.5, min_samples=2)
                labels = dbscan.fit_predict(vectors)
                cluster_method = "DBSCAN"
                
            else:
                return f"Unknown algorithm: {algorithm}. Use 'kmeans' or 'dbscan'."
            
            # Organize files by cluster
            for i, file_path in enumerate(file_paths):
                cluster_id = int(labels[i])
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(file_path)
            
            # Format the results
            output = f"## Code Clustering Results ({cluster_method})\n\n"
            output += f"Clustered {len(file_paths)} Python files into {len(clusters)} groups:\n\n"
            
            for cluster_id, files in sorted(clusters.items()):
                if cluster_id == -1:
                    output += "### Outliers (No Clear Cluster):\n"
                else:
                    output += f"### Cluster {cluster_id + 1}:\n"
                    
                output += f"Contains {len(files)} files:\n"
                for file in files[:10]:
                    output += f"- {file}\n"
                    
                if len(files) > 10:
                    output += f"... and {len(files) - 10} more\n"
                    
                output += "\n"
            
            return output
            
        except Exception as e:
            return f"Error during clustering: {str(e)}"

class SecurityAnalysisTool(BaseTool):
    """Tool for conducting security analysis of Python code."""
    
    name: str = "Security Analysis Tool"
    description: str = """
    Analyzes Python code for security vulnerabilities using:
    - Static code analysis
    - Pattern matching for known security issues
    - Risk scoring and prioritization
    
    Identifies potential security threats and provides recommendations.
    """

    def _run(self, file_patterns: str) -> str:
        """
        Analyze Python code for security vulnerabilities.
        
        Args:
            file_patterns: Glob patterns for Python files to analyze (comma-separated)
            
        Returns:
            A summary of security analysis results with prioritized recommendations
        """
        # This would typically use a security scanner like Bandit
        # For this example, we'll implement a simplified pattern-based analysis

        # Common security patterns to look for
        security_patterns = {
            "hardcoded_password": (r'password\s*=\s*[\'"][^\'"]+[\'"]', "Hardcoded password", "HIGH"),
            "sql_injection": (r'execute\([\'"].*\%.*[\'"]', "Potential SQL injection", "HIGH"),
            "command_injection": (r'os\.system\(|subprocess\.call\(|eval\(', "Potential command injection", "HIGH"),
            "insecure_pickle": (r'pickle\.loads\(|pickle\.load\(', "Insecure deserialization with pickle", "MEDIUM"),
            "weak_crypto": (r'md5|SHA1', "Weak cryptographic algorithm", "MEDIUM"),
            "logging_sensitive": (r'logging\.[a-z]+\(.*password', "Sensitive data in logs", "MEDIUM"),
            "assert_usage": (r'^assert\s', "Assert statements can be bypassed with -O flag", "LOW"),
            "yaml_load": (r'yaml\.load\((?!.*Loader)', "Insecure YAML loading without SafeLoader", "MEDIUM"),
            "jwt_verify": (r'jwt\.decode\(.*verify\s*=\s*False', "JWT signature verification disabled", "HIGH"),
            "debug_enabled": (r'DEBUG\s*=\s*True', "Debug mode enabled in production code", "LOW"),
        }

        import re

        patterns = [p.strip() for p in file_patterns.split(',')]
        results = []

        # Analyze each file
        for pattern in patterns:
            for file_path in glob.glob(pattern, recursive=True):
                if not file_path.endswith('.py'):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')

                    file_issues = []

                    # Check each pattern against the file
                    for regex, description, severity in security_patterns.values():
                        file_issues.extend(
                            {
                                "issue": description,
                                "severity": severity,
                                "line": i + 1,
                                "code": line.strip(),
                            }
                            for i, line in enumerate(lines)
                            if re.search(regex, line)
                        )
                    if file_issues:
                        results.append({
                            "file": file_path,
                            "issues": file_issues,
                            "total_issues": len(file_issues)
                        })

                except Exception as e:
                    results.append({
                        "file": file_path,
                        "error": str(e)
                    })

        # Sort results by total issues
        sorted_results = sorted(results, key=lambda x: x.get("total_issues", 0) if "total_issues" in x else 0, reverse=True)

        # Format the results
        output = "## Security Analysis Results\n\n"

        if not results:
            output += "No Python files matched the patterns or no security issues were found.\n"
            return output

        vulnerable_files = [r for r in results if "issues" in r and r["issues"]]
        output += f"Analyzed {len(results)} Python files. Found security issues in {len(vulnerable_files)} files:\n\n"

        # Group issues by severity
        severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for result in vulnerable_files:
            for issue in result["issues"]:
                severity_counts[issue["severity"]] += 1

        output += "### Issue Summary\n"
        output += f"- HIGH severity issues: {severity_counts['HIGH']}\n"
        output += f"- MEDIUM severity issues: {severity_counts['MEDIUM']}\n"
        output += f"- LOW severity issues: {severity_counts['LOW']}\n\n"

        # Show detailed results for files with issues
        for result in sorted_results[:10]:
            if "error" in result:
                output += f"### Error analyzing {result['file']}\n"
                output += f"Error: {result['error']}\n\n"
                continue

            if "issues" not in result or not result["issues"]:
                continue

            output += f"### {result['file']}\n"
            output += f"Found {len(result['issues'])} potential security issues:\n\n"

            # Group by severity
            for severity in ["HIGH", "MEDIUM", "LOW"]:
                severity_issues = [i for i in result["issues"] if i["severity"] == severity]
                if not severity_issues:
                    continue

                output += f"#### {severity} Severity Issues:\n"
                for issue in severity_issues:
                    output += f"- Line {issue['line']}: {issue['issue']}\n"
                    output += f"  `{issue['code']}`\n"
                output += "\n"

        # Add recommendations
        output += "### Recommendations\n"
        if severity_counts["HIGH"] > 0:
            output += "- **Critical**: Address all HIGH severity issues immediately\n"
        if severity_counts["MEDIUM"] > 0:
            output += "- **Important**: Review and fix MEDIUM severity issues as part of the next release\n"
        if severity_counts["LOW"] > 0:
            output += "- **Consider**: Address LOW severity issues when refactoring the affected code\n"

        return output

class PerformanceAnalysisTool(BaseTool):
    """Tool for analyzing code performance."""
    
    name: str = "Performance Analysis Tool"
    description: str = """
    Analyzes Python code for performance bottlenecks using:
    - Static analysis to identify potentially inefficient patterns
    - Big-O complexity estimation for functions
    - Performance anti-pattern detection
    
    Suggests optimization opportunities for improving code efficiency.
    """

    def _run(self, file_patterns: str) -> str:
        """
        Analyze Python code for performance bottlenecks and anti-patterns.
        
        Args:
            file_patterns: Glob patterns for Python files to analyze (comma-separated)
            
        Returns:
            A summary of performance analysis with recommendations
        """
        # Performance anti-patterns to detect
        perf_patterns = {
            "list_in_loop": (r'for .+ in .+:\s*\n\s+.+\.append\(', 
                             "List construction in loop - consider list comprehension", 
                             "MEDIUM"),
            "nested_loops": (r'for .+ in .+:(?:[^\n]+\n)+?[ \t]+for .+ in .+:', 
                            "Nested loops - potential O(n²) complexity", 
                            "HIGH"),
            "string_concat_loop": (r'for .+ in .+:\s*\n\s+.+\s*\+=\s*[\'"].+[\'"]', 
                                 "String concatenation in loop - use join() instead", 
                                 "MEDIUM"),
            "dict_lookup_loop": (r'for .+ in .+:\s*\n\s+if\s+.+\s+in\s+.+:', 
                                "Dictionary lookup in loop - potential O(n²) operation", 
                                "MEDIUM"),
            "global_var_loop": (r'global\s+.+\s*\n.+for\s+.+\s+in\s+.+:', 
                               "Global variable modified in loop", 
                               "LOW"),
            "deep_copy": (r'copy\.deepcopy\(', 
                         "Deep copy operation - can be expensive for large objects", 
                         "LOW"),
            "large_memory": (r'\.to_numpy\(\)|np\.array\(.*\)|\.values', 
                            "Converting to NumPy array - potential memory issue with large data", 
                            "MEDIUM"),
            "pandas_apply": (r'\.apply\(', 
                            "Pandas apply() - consider vectorized operations", 
                            "MEDIUM"),
            "recursive_calls": (r'def\s+([a-zA-Z0-9_]+).*\n(?:[^\n]+\n)*?[ \t]+\1\(', 
                               "Recursive function - watch for stack overflow with deep recursion", 
                               "LOW"),
            "regex_in_loop": (r'for .+ in .+:\s*\n\s+re\.(search|match|findall)\(', 
                             "Regular expression in loop - compile outside loop", 
                             "MEDIUM"),
        }

        import re

        patterns = [p.strip() for p in file_patterns.split(',')]
        results = []

        # Analyze each file
        for pattern in patterns:
            for file_path in glob.glob(pattern, recursive=True):
                if not file_path.endswith('.py'):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    file_issues = []

                    # Check each pattern against the file
                    for regex, description, severity in perf_patterns.values():
                        matches = re.finditer(regex, content, re.MULTILINE)
                        for match in matches:
                            line_no = content[:match.start()].count('\n') + 1
                            context = content[match.start():match.end()].strip()
                            if len(context) > 100:
                                context = context[:97] + "..."

                            file_issues.append({
                                "issue": description,
                                "severity": severity,
                                "line": line_no,
                                "context": context,
                            })

                    # Check for large functions (high cyclomatic complexity)
                    try:
                        cc_results = radon_cc.cc_visit(content)
                        file_issues.extend(
                            {
                                "issue": f"Function '{func.name}' has high cyclomatic complexity ({func.complexity})",
                                "severity": (
                                    "HIGH" if func.complexity > 20 else "MEDIUM"
                                ),
                                "line": func.lineno,
                                "context": f"Complexity: {func.complexity}",
                            }
                            for func in cc_results
                            if func.complexity > 10
                        )
                    except Exception:
                        pass

                    if file_issues:
                        results.append({
                            "file": file_path,
                            "issues": file_issues,
                            "total_issues": len(file_issues)
                        })

                except Exception as e:
                    results.append({
                        "file": file_path,
                        "error": str(e)
                    })

        # Sort results by total issues
        sorted_results = sorted(results, key=lambda x: x.get("total_issues", 0) if "total_issues" in x else 0, reverse=True)

        # Format the results
        output = "## Performance Analysis Results\n\n"

        if not results:
            output += "No Python files matched the patterns or no performance issues were found.\n"
            return output

        issue_files = [r for r in results if "issues" in r and r["issues"]]
        output += f"Analyzed {len(results)} Python files. Found performance concerns in {len(issue_files)} files:\n\n"

        # Group issues by severity
        severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for result in issue_files:
            for issue in result["issues"]:
                severity_counts[issue["severity"]] += 1

        output += "### Issue Summary\n"
        output += f"- HIGH impact issues: {severity_counts['HIGH']}\n"
        output += f"- MEDIUM impact issues: {severity_counts['MEDIUM']}\n"
        output += f"- LOW impact issues: {severity_counts['LOW']}\n\n"

        # Show detailed results for files with issues
        for result in sorted_results[:10]:
            if "error" in result:
                output += f"### Error analyzing {result['file']}\n"
                output += f"Error: {result['error']}\n\n"
                continue

            if "issues" not in result or not result["issues"]:
                continue

            output += f"### {result['file']}\n"
            output += f"Found {len(result['issues'])} potential performance issues:\n\n"

            # Group by severity
            for severity in ["HIGH", "MEDIUM", "LOW"]:
                severity_issues = [i for i in result["issues"] if i["severity"] == severity]
                if not severity_issues:
                    continue

                output += f"#### {severity} Impact Issues:\n"
                for issue in severity_issues:
                    output += f"- Line {issue['line']}: {issue['issue']}\n"
                    if "context" in issue:
                        output += f"  Context: `{issue['context']}`\n"
                output += "\n"

        # Add optimization recommendations
        output += "### Optimization Recommendations\n"

        if severity_counts["HIGH"] > 0:
            output += "- **Critical Optimizations**: Address O(n²) operations and high complexity functions first\n"
        if severity_counts["MEDIUM"] > 0:
            output += "- **Moderate Improvements**: Replace inefficient patterns with pythonic alternatives\n"
        if severity_counts["LOW"] > 0:
            output += "- **Fine-tuning**: Consider minor optimizations when working on the affected code\n"

        output += "\n### General Performance Best Practices\n"
        output += "- Use list/dict comprehensions instead of building collections in loops\n"
        output += "- Replace nested loops with more efficient data structures when possible\n"
        output += "- Consider NumPy vectorized operations for numerical processing\n"
        output += "- Profile your code to identify actual bottlenecks before optimizing\n"

        return output

class DocumentationAnalysisTool(BaseTool):
    """Tool for analyzing and improving code documentation."""
    
    name: str = "Documentation Analysis Tool"
    description: str = """
    Analyzes Python code documentation quality using NLP techniques:
    - Docstring completeness check
    - Function-to-documentation alignment analysis
    - Documentation gaps identification
    
    Helps improve and maintain comprehensive code documentation.
    """

    def _run(self, file_patterns: str) -> str:
        """
        Analyze documentation quality in Python files.
        
        Args:
            file_patterns: Glob patterns for Python files to analyze (comma-separated)
            
        Returns:
            A summary of documentation analysis with improvement suggestions
        """
        patterns = [p.strip() for p in file_patterns.split(',')]
        results = []

        class NodeVisitor(ast.NodeVisitor):
            def __init__(self):
                self.parent_map = {}

            def visit(self, node):
                for child in ast.iter_child_nodes(node):
                    self.parent_map[child] = node
                    self.visit(child)

        # Helper function to extract docstrings
        def extract_docstrings(tree):
            """Extract docstrings from an AST."""
            docstrings = []

            # First add parent references to all nodes
            visitor = NodeVisitor()
            visitor.visit(tree)
            parent_map = visitor.parent_map

            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                    # Check for module docstring
                    if isinstance(node, ast.Module) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                        docstrings.append(("module", node.body[0].value.s))

                    # Check for class/function docstring
                    elif hasattr(node, 'body') and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                        if isinstance(node, ast.ClassDef):
                            docstrings.append(("class", node.name, node.body[0].value.s))
                        elif isinstance(node, ast.FunctionDef):
                            docstrings.append(("function", node.name, node.body[0].value.s))

            return docstrings, parent_map

        # Analyze each file
        for pattern in patterns:
            for file_path in glob.glob(pattern, recursive=True):
                if not file_path.endswith('.py'):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Parse the AST
                    tree = ast.parse(content)

                    # Extract class and function definitions
                    classes = []
                    functions = []
                    module_docstring = None

                    # Extract docstrings with parent information
                    docstrings, parent_map = extract_docstrings(tree)

                    # Look for top-level classes and functions
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            classes.append(node)
                        elif isinstance(node, ast.FunctionDef):
                            # Only capture top-level functions (parent is Module)
                            if node in parent_map and isinstance(parent_map[node], ast.Module):
                                functions.append(node)
                        elif (isinstance(node, ast.Expr) and 
                              isinstance(node.value, ast.Str) and
                              node in parent_map and
                              isinstance(parent_map[node], ast.Module) and
                              parent_map[node].body and parent_map[node].body[0] == node):
                            module_docstring = node.value.s

                    # Analyze documentation quality
                    doc_issues = []

                    # Check module docstring
                    if not module_docstring:
                        doc_issues.append({
                            "type": "missing",
                            "element": "module",
                            "name": os.path.basename(file_path),
                            "issue": "Missing module-level docstring"
                        })

                    # Check class and function docstrings
                    documented_elements = {}
                    for doc_type, name, docstring in [d for d in docstrings if len(d) == 3]:
                        documented_elements[(doc_type, name)] = docstring

                        # Check docstring quality
                        if len(docstring.strip()) < 10:
                            doc_issues.append({
                                "type": "poor",
                                "element": doc_type,
                                "name": name,
                                "issue": f"Very short {doc_type} docstring (less than 10 chars)"
                            })

                        # Check if parameters are documented (for functions)
                        if doc_type == "function" and "param" not in docstring and "parameter" not in docstring.lower():
                            doc_issues.append({
                                "type": "incomplete",
                                "element": "function",
                                "name": name,
                                "issue": "No parameter documentation found in docstring"
                            })

                        # Check if return value is documented (for functions)
                        if doc_type == "function" and "return" not in docstring.lower():
                            doc_issues.append({
                                "type": "incomplete",
                                "element": "function",
                                "name": name,
                                "issue": "No return value documentation found in docstring"
                            })

                    # Check for missing docstrings
                    for cls in classes:
                        if ("class", cls.name) not in documented_elements:
                            doc_issues.append({
                                "type": "missing",
                                "element": "class",
                                "name": cls.name,
                                "issue": "Missing class docstring"
                            })

                    for func in functions:
                        if ("function", func.name) not in documented_elements:
                            doc_issues.append({
                                "type": "missing",
                                "element": "function",
                                "name": func.name,
                                "issue": "Missing function docstring"
                            })

                    if doc_issues:
                        results.append(
                            {
                                "file": file_path,
                                "issues": doc_issues,
                                "stats": {
                                    "total_classes": len(classes),
                                    "total_functions": len(functions),
                                    "documented_classes": sum(
                                        k[0] == "class"
                                        for k in documented_elements
                                    ),
                                    "documented_functions": sum(
                                        k[0] == "function"
                                        for k in documented_elements
                                    ),
                                    "has_module_docstring": module_docstring
                                    is not None,
                                },
                            }
                        )

                except Exception as e:
                    results.append({
                        "file": file_path,
                        "error": str(e)
                    })

        # Format the results
        output = "## Documentation Analysis Results\n\n"

        if not results:
            output += "No Python files matched the patterns or no documentation issues were found.\n"
            return output

        issue_files = [r for r in results if "issues" in r and r["issues"]]
        output += f"Analyzed {len(results)} Python files. Found documentation issues in {len(issue_files)} files.\n\n"

        # Calculate overall statistics
        total_classes = sum(r["stats"]["total_classes"] for r in results if "stats" in r)
        total_functions = sum(r["stats"]["total_functions"] for r in results if "stats" in r)
        documented_classes = sum(r["stats"]["documented_classes"] for r in results if "stats" in r)
        documented_functions = sum(r["stats"]["documented_functions"] for r in results if "stats" in r)
        files_with_module_docs = sum(bool("stats" in r and r["stats"]["has_module_docstring"])
                                 for r in results)

        output += "### Overall Documentation Statistics\n"
        output += f"- Files with module docstrings: {files_with_module_docs}/{len(results)} ({files_with_module_docs/len(results)*100:.1f}%)\n"
        output += f"- Classes with docstrings: {documented_classes}/{total_classes} ({documented_classes/total_classes*100:.1f}% if total_classes > 0 else 'N/A')\n"
        output += f"- Functions with docstrings: {documented_functions}/{total_functions} ({documented_functions/total_functions*100:.1f}% if total_functions > 0 else 'N/A')\n\n"

        # Group issues by type
        missing_count = sum(1 for r in issue_files for i in r["issues"] if i["type"] == "missing")
        poor_count = sum(1 for r in issue_files for i in r["issues"] if i["type"] == "poor")
        incomplete_count = sum(1 for r in issue_files for i in r["issues"] if i["type"] == "incomplete")

        output += "### Issue Summary\n"
        output += f"- Missing docstrings: {missing_count}\n"
        output += f"- Low-quality docstrings: {poor_count}\n"
        output += f"- Incomplete docstrings: {incomplete_count}\n\n"

        # Show detailed results for files with issues
        sorted_results = sorted(issue_files, key=lambda x: len(x["issues"]), reverse=True)

        for result in sorted_results[:5]:
            output += f"### {result['file']}\n"

            if "stats" in result:
                stats = result["stats"]
                output += "Documentation coverage:\n"
                output += f"- Module docstring: {'✅' if stats['has_module_docstring'] else '❌'}\n"
                output += f"- Classes: {stats['documented_classes']}/{stats['total_classes']} documented\n"
                output += f"- Functions: {stats['documented_functions']}/{stats['total_functions']} documented\n\n"

            output += "Issues:\n"

            # Group by issue type
            missing_issues = [i for i in result["issues"] if i["type"] == "missing"]
            poor_issues = [i for i in result["issues"] if i["type"] == "poor"]
            incomplete_issues = [i for i in result["issues"] if i["type"] == "incomplete"]

            if missing_issues:
                output += "#### Missing Docstrings:\n"
                for issue in missing_issues:
                    output += f"- {issue['element'].capitalize()} `{issue['name']}`: {issue['issue']}\n"
                output += "\n"

            if poor_issues:
                output += "#### Low-Quality Docstrings:\n"
                for issue in poor_issues:
                    output += f"- {issue['element'].capitalize()} `{issue['name']}`: {issue['issue']}\n"
                output += "\n"

            if incomplete_issues:
                output += "#### Incomplete Docstrings:\n"
                for issue in incomplete_issues:
                    output += f"- {issue['element'].capitalize()} `{issue['name']}`: {issue['issue']}\n"
                output += "\n"

        # Add recommendations
        output += "### Documentation Improvement Recommendations\n"
        output += "1. Add module-level docstrings to all Python files to describe their purpose\n"
        output += "2. Document all public classes and functions with descriptive docstrings\n"
        output += "3. Include parameter descriptions and return value information in function docstrings\n"
        output += "4. Use consistent docstring format (Google style, NumPy style, or reStructuredText)\n"
        output += "5. Consider using tools like Sphinx to generate documentation from docstrings\n"

        return output 