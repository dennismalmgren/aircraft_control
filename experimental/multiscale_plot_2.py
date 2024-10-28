import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def generate_hex_grid(scale, coverage_radius, area_size):
    """
    Generate a hexagonal grid of points scaled by a given factor.
    
    Parameters:
    - scale: Scaling factor for the grid layer.
    - coverage_radius: Radius of influence for each node.
    - area_size: Side length of the square area to cover with the grid.
    
    Returns:
    - points: Array of (x, y) coordinates for each node in the hexagonal grid.
    """
    # Hexagonal grid spacing (center-to-center distance of neighboring nodes)
    spacing = coverage_radius * scale * np.sqrt(3)
    # Generate points in a hexagonal grid pattern within the specified area
    points = []
    for x in np.arange(0, area_size, spacing):
        for y in np.arange(0, area_size, spacing * np.sqrt(3) / 2):
            offset_x = x + (spacing / 2) * ((y // (spacing * np.sqrt(3) / 2)) % 2)
            points.append((offset_x, y))
    return np.array(points)

def calculate_overlaps(layers, coverage_radius):
    """
    Calculate the number of overlaps between layers using coverage radius.
    
    Parameters:
    - layers: List of arrays, each containing (x, y) coordinates of nodes in a layer.
    - coverage_radius: Radius of influence for each node.
    
    Returns:
    - overlap_count: Total number of overlapping nodes across layers.
    """
    overlap_count = 0
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            tree_i = cKDTree(layers[i])
            tree_j = cKDTree(layers[j])
            overlap_count += len(tree_i.query_ball_tree(tree_j, coverage_radius))
    return overlap_count

def simulate_scale_factors(scale_factors, coverage_radius=1.0, area_size=20.0, N=3):
    """
    Simulate normalized overlaps for a range of scale factors across N layers of hexagonal grids.
    
    Parameters:
    - scale_factors: List of scale factors to test.
    - coverage_radius: Radius of influence for each node.
    - area_size: Side length of the square area to cover with the grid.
    - N: Number of layers to generate.
    
    Returns:
    - results: Dictionary with scale factors as keys and normalized overlap counts as values.
    """
    results = {}
    for scale_factor in scale_factors:
        layers = []
        total_nodes = 0  # Total node count across all layers
        for layer in range(N):
            scale = scale_factor ** layer
            points = generate_hex_grid(scale, coverage_radius, area_size)
            layers.append(points)
            total_nodes += len(points)  # Count nodes in each layer
        
        # Calculate the total number of overlaps
        overlap_count = calculate_overlaps(layers, coverage_radius)
        
        # Normalize overlaps by total node count
        normalized_overlap = overlap_count / total_nodes
        results[scale_factor] = normalized_overlap
        print(f"Scale Factor: {scale_factor:.3f}, Overlaps: {overlap_count}, "
              f"Total Nodes: {total_nodes}, Normalized Overlap: {normalized_overlap:.4f}")
    return results

# Define parameters
scale_factors = np.linspace(1.2, 1.7, 10)  # Scale factors to test
coverage_radius = 1.0  # Radius of influence for each node
area_size = 20.0  # Size of the area to tile with hexagonal grids
N = 3  # Number of grid layers

# Run simulation
results = simulate_scale_factors(scale_factors, coverage_radius, area_size, N)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(scale_factors, list(results.values()), marker='o', linestyle='-')
plt.xlabel("Scale Factor")
plt.ylabel("Normalized Overlap (Overlap / Total Nodes)")
plt.title("Normalized Overlap vs Scale Factor for N Grid Layers")
plt.grid()
plt.show()
