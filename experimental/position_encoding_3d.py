import numpy as np
import matplotlib.pyplot as plt

# Parameters
lambda_ = 1  # Wavelength for sinusoidal function
extent = 3 * lambda_  # Range for x, y, z directions

# Define grid for x, y, z
x = np.linspace(-extent, extent, 100)
y = np.linspace(-extent, extent, 100)
z = np.linspace(-extent, extent, 100)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Define wave vectors for HCP pattern in 3D
k1 = np.array([1, 0, 0])
k2 = np.array([-0.5, np.sqrt(3)/2, 0])
k3 = np.array([-0.5, -np.sqrt(3)/2, 0])
k4 = np.array([0, 0, 2*np.sqrt(6)/3])  # Adjusted for HCP vertical periodicity
k5 = np.array([1, 0, np.sqrt(6)/3])
k6 = np.array([-0.5, np.sqrt(3)/2, np.sqrt(6)/3])
k7 = np.array([-0.5, -np.sqrt(3)/2, np.sqrt(6)/3])

# Scale wave vectors by wavelength
factor = 2 * np.pi / lambda_
k_vectors = [factor * k for k in [k1, k2, k3, k4, k5, k6, k7]]

# Calculate the combined sinusoidal intensity map
pattern = sum(np.sin(k[0]*X + k[1]*Y + k[2]*Z) for k in k_vectors)

# Visualize a slice at the middle of Z-axis
z_index = len(z) // 2
plt.figure(figsize=(8, 8))
plt.imshow(pattern[:, :, z_index], extent=(-extent, extent, -extent, extent),
           cmap='viridis', origin='lower')
plt.colorbar(label='Intensity')
plt.title(f'HCP Sinusoidal Pattern at Z=0 (Slice)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
