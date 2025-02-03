import numpy as np
import statistics as st
arr=np.array([1,2,3,6,4,5])
# a.
x = arr[::-1]
print(x) 
# b.
# i.
x = np.array([1,2,3,4,5,1,2,1,1,1])
modeX=st.mode(x)
print("Most frequent value :",modeX)
print("index : ",np.where(x==modeX))
# ii.
y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3 ,3])
modeY=st.mode(y)
print("Most frequent value :",modeY)
print("index : ",np.where(y==modeY))
