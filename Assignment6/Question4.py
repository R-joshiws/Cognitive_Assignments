import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/AnjulaMehto/MCA/main/company_sales_data.csv"
df = pd.read_csv(url)

# Total Profit Line Plot
plt.figure(figsize=(10, 5))
sns.lineplot(x=df.index, y=df["total_profit"], marker="o")
plt.title("Total Profit Over Months")
plt.xlabel("Month")
plt.ylabel("Total Profit")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Product Sales Multiline Plot
plt.figure(figsize=(10, 5))
sales_columns = ['facecream', 'facewash', 'toothpaste', 'bathingsoap', 'shampoo', 'moisturizer']
for col in sales_columns:
    sns.lineplot(x=df.index, y=df[col], label=col)
plt.title("Product Sales Over Months")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Bar Chart for All Features
plt.figure(figsize=(12, 6))
df.plot(kind="bar", figsize=(12, 6))
plt.title("Bar Chart for All Features")
plt.xlabel("Month")
plt.ylabel("Values")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
