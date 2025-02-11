import numpy as np
#Initials are R and J 
X = ord('R') + ord('J') 

sales = np.array([X, X+50, X+100, X+150, X+200])
print("Original Sales:", sales)

# (b) Computing personalized tax rate
tax_rate = ((X % 5) + 5) / 100
taxed_sales = sales + sales * tax_rate
print("Taxed Sales:", taxed_sales)

# (c) Apply discount based on condition
discounted_sales = np.where(sales < X+100, sales * 0.95, sales * 0.90)
print("Discounted Sales:", discounted_sales)

# (d) Expand sales for multiple weeks
weekly_sales = np.vstack([sales, sales * 1.02, sales * 1.04])  # Increasing 2% per week

print("Weekly Sales:\n", weekly_sales)
