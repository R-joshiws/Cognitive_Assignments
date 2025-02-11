import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)

# Compute y values for different functions
y1 = x**2
y2 = np.sin(x)
y3 = np.exp(x)
y4 = np.log(np.abs(x) + 1)

# Plot each function
plt.figure(figsize=(10, 6))

plt.plot(x, y1, label="y = x^2")
plt.plot(x, y2, label="y = sin(x)")
plt.plot(x, y3, label="y = exp(x)")
plt.plot(x, y4, label="y = log(|x| + 1)")

# Add title, labels, and grid
plt.title("Function Plots using NumPy")
plt.xlabel("x values")
plt.ylabel("y values")
plt.legend()
plt.grid(True)

plt.show()
