import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) +'/..'))

import lib.CheeseBurger as cb

# feature 배열에서 entropy 를 계산하는것과,
# feature x class 행렬에서 entropy 를 계산하는 것에 차이가 있을 것 같다.
# 이를 비교.

# data_with_class = [
#     [10, 0],
#     [10, 0],
#     [10, 0],
#     [10, 0],
#     [10, 0],
#     [10, 0],
#     [10, 0],
#     [0, 10],
#     [0, 10],
#     [0, 10],
#     [0, 10],
#     [0, 10],
#     [0, 10],
#     [0, 10],
#     [5, 5]
# ]

data_with_class = [
    [10,0,0],
    [1,0,0],
    [10,0,0],
    [1,0,0],
    [5,1,4],
    [5,1,4],
    [5,1,4],
    [5,1,4],
]

data_ignore_class = list(map(lambda x: sum(x), data_with_class))


data_with_class1 = list(map(lambda x: x[0], data_with_class))


data_with_class2 = list(map(lambda x: x[1], data_with_class))


print(cb.Appetizer.entropy_list(cb, data_ignore_class))


print(cb.Appetizer.entropy_list(cb, data_with_class1))


print(cb.Appetizer.entropy_list(cb, data_with_class2))

def sumEntropyByClass(data:list):
    entropy_class_summary = 0
    for col in data:

        entropy_class_summary += cb.Appetizer.entropy_list(
            cb, 
            list(map(lambda x: x[col], data_with_class))
        )
    return entropy_class_summary

print(sumEntropyByClass(data_with_class))