import numpy as np
import matplotlib.pyplot as plt

M = float(input("Enter a value for M: "))
x = np.linspace(-10, 10, 400)
y1 = M * x**2
y2 = M * np.sin(x)

plt.plot(x, y1, label=f"y = {M} * x^2", color="blue", linestyle="-")
plt.plot(x, y2, label=f"y = {M} * sin(x)", color="red", linestyle="--")

plt.legend()
plt.grid(True)
plt.title("Plot of y = M * x^2 and y = M * sin(x)")
plt.xlabel("x")
plt.ylabel("y")

plt.show()
