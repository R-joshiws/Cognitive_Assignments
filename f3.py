import pandas as pd
df= pd.read_csv('/Users/yashagarwal/Downloads/employees.csv')
# print(df)

# a) Shape (number of rows and columns) of the DataFrame.
row, col = df.shape
print(row,col)

# b) Summary of the DataFrame that includes the data types and non-null counts for each column.
print(df.info())

# c) Generate descriptive statistics.
print(df.describe())

# d)
df.head(5)
df.tail(3)


#  e) Calculate the following statistics from the dataset:
# i. The average salary of employees.
# ii. The total bonus paid to all employees.
# iii. The youngest employee's age.
# iv. The highest performance rating.


print(df["Salary"].mean())
print(df["Bonus"].sum())
print(df.Age.min())
print(df["Rating"].max())


# f) Sort the DataFrame by the Salary column in descending order.
print(df.sort_values(by="Salary",ascending=False))

# g) Add a new column that categorizes employees based on their performance rating:
# i. Excellent for ratings >= 4.5
# ii. Good for ratings >= 4.0 but < 4.5
# iii. Average for ratings < 4.0
def cat_performance(rating):
    if rating >= 4.5:
        return "Excellent"
    elif rating >= 4.0:
        return "Good"
    else:
        return "Average"
df['Performance'] = df['Rating'].apply(cat_performance)



# h) Identify missing values in the DataFrame.
print(df.isna())

# i) Rename the Employee_ID column to ID.
df.rename(columns={"Employee_ID":"ID"},inplace=True)

# j) Find all employees who:
# i. Have more than 5 years of experience.
# ii. Belong to the IT department.
print(df.loc[df.Years_of_Experience>5])
print(df.loc[(df.Department=="IT")])

# k) Modify the dataset by adding a new column, Tax, which deducts 10% of the Salary.
def taxdeduce(salary):
    return salary*0.1
df["Tax"]=df["Salary"].apply(taxdeduce)
print(df)
# l) Save the modified DataFrame (with added columns) to a new CSV file.
df.to_csv('/Users/yashagarwal/Downloads/modified_employees.csv',index=False)
