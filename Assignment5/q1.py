import numpy as np

gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]')

# i. Sum of all elements
sum_all = np.sum(gfg)

# ii. Sum of all elements row-wise
sum_row_wise = np.sum(gfg, axis=1)

# iii. Sum of all elements column-wise
sum_col_wise = np.sum(gfg, axis=0)

print("Sum of all elements:", sum_all)
print("Row-wise sum:", sum_row_wise)
print("Column-wise sum:", sum_col_wise)
