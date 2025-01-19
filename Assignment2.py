import random
#QUESTION 1:
# i.
L=[10,20,30,40,50,60,70,80]
print(L)
L.extend([200,300])
print(L)
# ii.
L.remove(10)
L.remove(30)
print(L)
# iii.
L.sort()
print("Ascensing Order")
print(L)
# iv.
print("Descending Order")
L.sort(reverse = True)
print(L)

#QUESTION 2:
newTuple=(45,89.5,76,45.4,89,92,58,45)
# i.
maxEl=max(newTuple)
print(f"The MAX score is : { maxEl} at index : {newTuple.index(maxEl)}")
# ii.
minEl=min(newTuple)
print(f"The MIN score is : {minEl} ")
print(f"It's repeted {newTuple.count(minEl)} times ")
# iii.
result = list(reversed(newTuple))
print(result)
# iv.
if (76 in newTuple):
    print(f"76 is present at index {newTuple.index(76)}")
else:
    print("76 is not present in the tuple")
 
#QUESTION 3:
randomList=[random.randint(100,900) for _ in range(100)]
# i.
oddCount=0
print("ODD numbers: ")
for x in randomList:
    if x%2!=0:
        oddCount+=1
        print(x)
print("COunt of ODD numbers is :",oddCount)
# ii.
evenCount=0
for x in randomList:
    if x%2==0:
        print(x)
        evenCount+=1
print("COunt of EVEN numbers is :",evenCount)
# iii.
primeCount=0
for i in randomList:
        if i == 0 or i == 1:
            continue
        else:
            for j in range(2, int(i**0.5)+1):
                if i % j == 0:
                    break
            else:
                print(i)
                primeCount+=1
print("COunt of PRIME numbers is :",primeCount)

#QUESTION 4:
A={34,56,78,90}
B={78,45,90,23}
# i.
unique=A.union(B)
print("Unique scores are:")
print(unique)
# ii.
common=A.intersection(B)
print("Common scores to both teams are :")
print(common)
# iii.
diff=A.symmetric_difference(B)
print("Scores exclusive to teams are:")
print(diff)
# iv.
print("Are scores of A a subset of B :",A.issubset(B))
print("Are scores of B a superset of A :",B.issuperset(A))
# v.
X=int(input("Enter the score to remove : "))
if (X in A):
    A.remove(X)
    print(A)
else:
    print(f"{X} is not present")

#QUESTION 5:
sample_dict = {
    "name":"Kelly",
    "age":25,
    "salary":8000,
    "city":"New york"
}
print("OLD dictionary",sample_dict)
sample_dict['location'] = sample_dict.pop('city')
print("NEW dictionary",sample_dict)