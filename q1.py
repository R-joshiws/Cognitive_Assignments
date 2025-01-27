import pandas as pd;
data={
    "Tid":[1,2,3,4,5,6,7,8,9,10],
    "Refund":["Yes","No","No","Yes","No","No","Yes","No","No","Yes"],
    "Marital Status":["Single","Married","Single","Married","Divorced","Married","Divorced","Married","Single","Married"],
    "Taxable Income":["125k","100k","70K","120K","95K","60K","220K","85K","75k","90K"]
}
df=pd.DataFrame(data)

# From the above table that you have created, locate row 0, 4, 7 and 8 using DataFrame.
# print(df.loc[[0,4,7,8]])

# Q.3 Navigate the DataFrame and do the following task for the table created in question 1:


# 1. Select row from index 3 to 7.
# print(df.loc[3:8])

# 2. Select row from index 4 to 8, and column 2 to 4.
# print(df.iloc[3:8,2:5])

# # 3. Select all rows with column index 1 to 3 (include index 3 during selection).
# print(df.iloc[:,1:4])
# # or
# print(df[["Refund","Marital Status","Taxable Income"]])


