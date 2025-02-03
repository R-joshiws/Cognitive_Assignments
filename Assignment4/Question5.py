import numpy as np
ucs420_ramit=np.array([[10,20,30,40],[50,60,70,80],[90,15,20,35]])
import numpy as np

mean_value = np.mean(ucs420_ramit)
median_value = np.median(ucs420_ramit)
max_value = np.max(ucs420_ramit)
min_value = np.min(ucs420_ramit)
unique_elements = np.unique(ucs420_ramit)

# Print computed values
print("Original Array:\n", ucs420_ramit)
print("Mean:", mean_value)
print("Median:", median_value)
print("Max:", max_value)
print("Min:", min_value)
print("Unique Elements:", unique_elements)

reshaped_ucs420_ramit = ucs420_ramit.reshape(4, 3)
print("\nReshaped Array (4x3):\n", reshaped_ucs420_ramit)

resized_ucs420_ramit = np.resize(ucs420_ramit, (2, 3))
print("\nResized Array (2x3):\n", resized_ucs420_ramit)
