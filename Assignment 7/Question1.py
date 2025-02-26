import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
roll_number = 102317270
np.random.seed(roll_number)

sales_data = np.random.randint(1000, 5001, size=(12, 4))

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
columns = ['Electronics', 'Clothing', 'Home & Kitchen', 'Sports']

sales_df = pd.DataFrame(sales_data, index=months, columns=columns)
sales_df.head()
sales_df.describe()

total_sales_per_category = sales_df.sum()
total_sales_per_month = sales_df.sum(axis=1)
sales_df['Total Sales'] = total_sales_per_month


sales_df.head()

sales_df['Growth Rate'] = sales_df['Total Sales'].pct_change() * 100


sales_df.head()

if roll_number % 2 == 0:

    sales_df['Electronics Discounted'] = sales_df['Electronics'] * 0.90
else:

    sales_df['Clothing Discounted'] = sales_df['Clothing'] * 0.85
sales_df.head()


plt.figure(figsize=(10, 6))
for column in sales_df.columns[:-2]:
    plt.plot(sales_df.index, sales_df[column], label=column)

plt.title('Monthly Sales Trends for Each Category')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=sales_df.drop(columns=['Total Sales', 'Growth Rate']), orient='v')
plt.title('Sales Distribution by Category')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
