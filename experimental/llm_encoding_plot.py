import numpy as np
import matplotlib.pyplot as plt

# Parameters
d_model = 3           # Embedding dimension
max_pos = 50          # Number of positions to plot
context_length = 50   # Maximum context length (sets the highest frequency limit)

# Positional encoding function with proper frequency scaling
def positional_encoding(pos, d_model, context_length):
    encoding = np.zeros((pos, d_model))
    for p in range(pos):
        for i in range(d_model):
            # Adjust the frequency scaling to avoid near-constant values
            # Frequency scaling now completes <2 cycles within the context length
            frequency = context_length / (2 ** ((i + 1) / d_model))
            if i % 2 == 0:
                encoding[p, i] = np.sin(p / frequency)
            else:
                encoding[p, i] = np.cos(p / frequency)
    return encoding

# Calculate positional encodings
positions = np.arange(max_pos)
encodings = positional_encoding(max_pos, d_model, context_length)

# Plot each dimension's encoding as a separate line
plt.figure(figsize=(12, 6))
for i in range(d_model):
    plt.plot(positions, encodings[:, i], label=f'Dimension {i+1}')

# Labels and legends
plt.title("Sinusoidal Positional Encoding (d_model=3) with Adjusted Frequency Scaling")
plt.xlabel("Position")
plt.ylabel("Encoding Value")
plt.legend()
plt.grid()
plt.show()
