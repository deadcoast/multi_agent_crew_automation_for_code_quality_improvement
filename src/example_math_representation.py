#!/usr/bin/env python
"""
Simplified example to demonstrate the 3D visualization of algorithm quality assessment.
"""

import os
import sys

import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create simple 3D visualization function
def visualize_algorithms_3d():
    """Create a simple 3D visualization of algorithm quality."""
    # Create sample data points
    algorithms = {
        "merge_sort": [0.1, 0.2, 0.3],       # Good algorithm
        "quicksort": [0.15, 0.25, 0.35],     # Good algorithm
        "insertion_sort": [0.3, 0.4, 0.2],   # Medium algorithm
        "bubble_sort": [0.8, 0.7, 0.9]       # Poor algorithm
    }
    
    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color mapping for different types of points
    colors = {
        "merge_sort": "blue",
        "quicksort": "green",
        "insertion_sort": "orange",
        "bubble_sort": "red"
    }
    
    # Plot each point
    for name, coords in algorithms.items():
        color = colors.get(name, "gray")
        marker = '*' if name == "quicksort" else 'o'
        size = 200 if name == "quicksort" else 100
        ax.scatter(coords[0], coords[1], coords[2], color=color, s=size, marker=marker, label=name)
    
    # Add labels and legend
    ax.set_xlabel('Time Complexity')
    ax.set_ylabel('Space Efficiency')
    ax.set_zlabel('Code Clarity')
    ax.set_title('Algorithm Quality Assessment in 3D Space')
    ax.legend()
    
    # Save the figure with absolute path
    output_path = os.path.join(os.getcwd(), "algorithm_quality_3d.png")
    plt.savefig(output_path)
    plt.close()
    
    print(f"3D visualization saved to: {output_path}")
    return output_path

def main():
    """Main function to demonstrate 3D visualization."""
    print("=" * 80)
    print("SIMPLIFIED 3D ALGORITHM VISUALIZATION EXAMPLE")
    print("=" * 80)
    
    # Generate and save 3D visualization
    output_path = visualize_algorithms_3d()
    
    # Verify file was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"Visualization file created successfully ({file_size} bytes)")
    else:
        print("Error: Visualization file was not created")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 