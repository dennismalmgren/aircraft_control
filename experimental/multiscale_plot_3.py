import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def generate_hex_grid(scale, area_size):
    """
    Generate a hexagonal grid of points at a given scale within a square area.
    
    Parameters:
    - scale: Scaling factor for the grid layer.
    - area_size: Side length of the square area to cover with the grid.
    
    Returns:
    - points: Array of (x, y) coordinates for each node in the hexagonal grid.
    """
    spacing = scale * np.sqrt(3)  # Center-to-center distance in hexagonal grid
    points = []
    for x in np.arange(0, area_size, spacing):
        for y in np.arange(0, area_size, spacing * np.sqrt(3) / 2):
            offset_x = x + (spacing / 2) * ((y // (spacing * np.sqrt(3) / 2)) % 2)
            points.append((offset_x, y))
    return np.array(points)

def count_unique_positions(layers, threshold=0.7):
    """
    Count the number of unique positions across multiple layers.
    
    Parameters:
    - layers: List of arrays, each containing (x, y) coordinates of nodes in a layer.
    - threshold: Distance threshold to consider points as overlapping.
    
    Returns:
    - unique_count: Number of unique positions across all layers.
    """
    all_points = np.vstack(layers)
    tree = cKDTree(all_points)
    unique_points = tree.query_ball_tree(tree, threshold)
    
    # Count unique points by filtering out duplicates
    unique_set = set()
    for indices in unique_points:
        unique_set.add(tuple(all_points[indices[0]]))  # Use the first point in each group as unique
    return len(unique_set)

def simulate_positional_uniqueness(scale_factors, area_size=20.0, N=3):
    """
    Simulate positional uniqueness for a range of scale factors across N layers of hexagonal grids.
    
    Parameters:
    - scale_factors: List of scale factors to test.
    - area_size: Side length of the square area to tile with hexagonal grids.
    - N: Number of layers to generate.
    
    Returns:
    - results: Dictionary with scale factors as keys and unique position counts as values.
    """
    results = {}
    for scale_factor in scale_factors:
        layers = []
        for layer in range(N):
            # Scale each layer based on the scale factor and layer index
            scale = scale_factor ** layer
            points = generate_hex_grid(scale, area_size)
            layers.append(points)
        
        # Calculate the number of unique positions
        unique_count = count_unique_positions(layers)
        results[scale_factor] = unique_count
        print(f"Scale Factor: {scale_factor:.3f}, Unique Positions: {unique_count}")
    
    return results

# Define parameters
scale_factors = np.linspace(1.2, 1.7, 10)  # Scale factors to test
area_size = 20.0  # Size of the area to tile with hexagonal grids
N = 3  # Number of grid layers

# Run simulation
results = simulate_positional_uniqueness(scale_factors, area_size, N)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(scale_factors, list(results.values()), marker='o', linestyle='-')
plt.xlabel("Scale Factor")
plt.ylabel("Unique Positions")
plt.title("Unique Position Count vs Scale Factor for N Grid Layers")
plt.grid()
plt.show()
