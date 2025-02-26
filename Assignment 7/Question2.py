import numpy as np
array = np.array([[1, -2, 3], [-4, 5, -6]])
abs_array = np.abs(array)
print("Element-wise absolute value:")
print(abs_array)

flattened_array = array.flatten()

percentiles_flattened = np.percentile(flattened_array, [25, 50, 75])
print("\nPercentiles (25th, 50th, 75th) for flattened array:")
print(f"25th: {percentiles_flattened[0]}, 50th: {percentiles_flattened[1]}, 75th: {percentiles_flattened[2]}")

percentiles_columns = np.percentile(array, [25, 50, 75], axis=0)
print("\nPercentiles (25th, 50th, 75th) for each column:")
print(f"25th: {percentiles_columns[0]}")
print(f"50th: {percentiles_columns[1]}")
print(f"75th: {percentiles_columns[2]}")

percentiles_rows = np.percentile(array, [25, 50, 75], axis=1)
print("\nPercentiles (25th, 50th, 75th) for each row:")
print(f"25th: {percentiles_rows[0]}")
print(f"50th: {percentiles_rows[1]}")
print(f"75th: {percentiles_rows[2]}")

mean_flattened = np.mean(flattened_array)
median_flattened = np.median(flattened_array)
std_flattened = np.std(flattened_array)

print("\nMean, Median, and Standard Deviation for flattened array:")
print(f"Mean: {mean_flattened}, Median: {median_flattened}, Std Dev: {std_flattened}")

mean_columns = np.mean(array, axis=0)
median_columns = np.median(array, axis=0)
std_columns = np.std(array, axis=0)

print("\nMean, Median, and Standard Deviation for each column:")
print(f"Mean: {mean_columns}")
print(f"Median: {median_columns}")
print(f"Std Dev: {std_columns}")

mean_rows = np.mean(array, axis=1)
median_rows = np.median(array, axis=1)
std_rows = np.std(array, axis=1)

print("\nMean, Median, and Standard Deviation for each row:")
print(f"Mean: {mean_rows}")
print(f"Median: {median_rows}")
print(f"Std Dev: {std_rows}")
