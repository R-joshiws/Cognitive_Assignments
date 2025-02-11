import numpy as np
# (a) 
array_a = np.array([10, 52, 62, 16, 16, 54, 453])

# i. Sorted array
sorted_array = np.sort(array_a)

# ii. Indices of sorted array
sorted_indices = np.argsort(array_a)

# iii. 4 smallest elements
smallest_4 = np.sort(array_a)[:4]

# iv. 5 largest elements
largest_5 = np.sort(array_a)[-5:]

print("Sorted array:", sorted_array)
print("Indices of sorted array:", sorted_indices)
print("4 smallest elements are :", smallest_4)
print("5 largest elements are:", largest_5)

# (b) Given array
array_b = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])

# i. Integer elements only
int_elements = array_b[array_b == array_b.astype(int)]

# ii. Float elements only
float_elements = array_b[array_b != array_b.astype(int)]

print("Integer elements:", int_elements)
print("Float elements:", float_elements)
