import numpy as np

ramit = np.linspace(10, 100, 25)

print("Dimensions:", ramit.ndim)
print("Shape:", ramit.shape)
print("Total elements:", ramit.size)
print("Data type of elements:", ramit.dtype)
print("Total bytes consumed:", ramit.nbytes)

transpose = ramit.reshape(25, 1)  
print("\nTransposed array:\n", transpose)

transpose_T = ramit.T
print("\nTransposed array using T attribute :", transpose_T)
