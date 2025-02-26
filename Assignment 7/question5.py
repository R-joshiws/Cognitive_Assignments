def swap_elements_in_set(my_set, index1, index2):
    temp_list = list(my_set)

    if index1 < len(temp_list) and index2 < len(temp_list):
        temp_list[index1], temp_list[index2] = temp_list[index2], temp_list[index1]


        swapped_set = set(temp_list)
        return swapped_set
    else:
        return "Indices are out of range."
my_set = {10, 20, 30, 40, 50}
index1 = 1
index2 = 3

swapped_set = swap_elements_in_set(my_set, index1, index2)
print("Set after swapping:", swapped_set)
