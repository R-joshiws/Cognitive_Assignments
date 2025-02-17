import numpy as np
import matplotlib.pyplot as plt

roll_number = 123456  # Replace with your actual roll number
np.random.seed(roll_number)

data = np.random.randn(50)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(np.cumsum(data), color='b')
axes[0, 0].set_title("Cumulative Sum")
axes[0, 0].set_xlabel("Index")
axes[0, 0].set_ylabel("Sum")
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

axes[0, 1].scatter(range(50), data, color='r', alpha=0.7)
axes[0, 1].set_title("Random Noise")
axes[0, 1].set_xlabel("Index")
axes[0, 1].set_ylabel("Value")
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

fig.delaxes(axes[1, 0])  
fig.delaxes(axes[1, 1])  

plt.tight_layout()
plt.show()
