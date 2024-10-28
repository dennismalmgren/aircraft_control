import numpy as np
import matplotlib.pyplot as plt

# Parameters
lambda_ = 1  # Wavelength for sinusoidal function
extent = 3 * lambda_  # Range for x and y directions (3 times lambda)

# Define grid for x and y
x = np.linspace(-extent, extent, 200)
y = np.linspace(-extent, extent, 200)
X, Y = np.meshgrid(x, y)

# Define wave vectors for hexagonal pattern
k1 = np.array([1, 0])
k2 = np.array([-0.5, np.sqrt(3) / 2])
k3 = np.array([-0.5, -np.sqrt(3) / 2])

# Calculate sinusoidal pattern function for hexagonal grid
def hexagonal_pattern(X, Y, lambda_):
    # Scale wave vectors by wavelength
    factor = 2 * np.pi / lambda_
    k1_scaled, k2_scaled, k3_scaled = factor * k1, factor * k2, factor * k3

    # Calculate the combined sinusoidal intensity map
    pattern = (np.sin(k1_scaled[0] * X + k1_scaled[1] * Y) +
               np.sin(k2_scaled[0] * X + k2_scaled[1] * Y) +
               np.sin(k3_scaled[0] * X + k3_scaled[1] * Y))
    return pattern

# Compute the pattern
pattern = hexagonal_pattern(X, Y, lambda_)

# Plotting the intensity map
plt.figure(figsize=(8, 8))
plt.imshow(pattern, extent=(-extent, extent, -extent, extent), cmap='viridis', origin='lower')
plt.colorbar(label='Intensity')
plt.title(f'Hexagonal Sinusoidal Pattern (3λ x 3λ region)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
