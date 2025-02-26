def swap_elements(lst, index1, index2):

    if index1 < len(lst) and index2 < len(lst):

        temp = lst[index1]
        lst[index1] = lst[index2]
        lst[index2] = temp
        return lst
    else:
        return "Indices are out of range."
my_list = [10, 20, 30, 40, 50]

index1 = 1
index2 = 3
swapped_list = swap_elements(my_list, index1, index2)
print("List after swapping:", swapped_list)
