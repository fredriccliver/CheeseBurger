# feature 배열에서 entropy 를 계산하는것과,
# feature x class 행렬에서 entropy 를 계산하는 것에 차이가 있을 것 같다.
# 이를 비교.

import lib.CheeseBurger as cb

data_with_class = [
    [10, 0],
    [10, 0],
    [10, 0],
    [10, 0],
    [10, 0],
    [10, 0],
    [10, 0],
    [0, 10],
    [0, 10],
    [0, 10],
    [0, 10],
    [0, 10],
    [0, 10],
    [0, 10],
    [5, 5]
]



data_ignore_class = list(map(lambda x: sum(x), data_with_class))
data_with_class1 = list(map(lambda x: x[0], data_with_class))
data_with_class2 = list(map(lambda x: x[1], data_with_class))

print(data_ignore_class)
print(data_with_class1)
print(data_with_class2)
