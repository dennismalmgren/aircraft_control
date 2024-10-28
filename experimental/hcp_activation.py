import numpy as np



def activation(x, y, z, lambda_):
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
    activation = sum(np.sin(k[0]*x + k[1]*y + k[2]*z) for k in k_vectors)
    return (1/7) * activation


a1 = activation(0.5, 0.5, 0, 1)
a2 = activation(0.5, 0.5, 0, 1.5)
print(a1)
print(a2)
