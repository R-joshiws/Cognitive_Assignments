import numpy as np

# Define the 2D array
arr = np.array([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])

# a)
element_a = arr[0, 1] 
print("Element at 1st row, 2nd column:", element_a)

# b) 
element_b = arr[2, 0] 
print("Element at 3rd row, 1st column:", element_b)
