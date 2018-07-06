import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) +'/..'))

import lib.CheeseBurger as cb
import pandas as pd

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

# data_ignore_class = list(map(lambda x: sum(x), data_with_class))

# data_with_class1 = list(map(lambda x: x[0], data_with_class))
# data_with_class2 = list(map(lambda x: x[1], data_with_class))
# data_with_class3 = list(map(lambda x: x[1], data_with_class))

# print("entropy class ignored")
# print(cb.Appetizer.entropy_list(cb, data_ignore_class))

# print("\n")

# print(cb.Appetizer.entropy_list(cb, data_with_class1))
# print(cb.Appetizer.entropy_list(cb, data_with_class2))
# print(cb.Appetizer.entropy_list(cb, data_with_class3))

class_size = len(data_with_class[0])

def sumEntropyByClass(data:list):
    entropy_class_summary = 0
    
    for col in range(class_size):
        
        entropy_class_summary += cb.Appetizer.entropy_from_list(
            cb, 
            list(map(lambda x: x[col], data_with_class))
        )
    return entropy_class_summary

# print("\n")
# print("total Entropy for all class")
# print(sumEntropyByClass(data_with_class))

train = pd.read_csv('data/train.csv')
label_col_name = 'Survived'


def entropy_independently(train_data: list, label_col_name: str, feature_col_name: str):
    
    class_names = list(train_data[[label_col_name]].groupby(label_col_name).size().keys())
    divided_col_by_classes = []

    for class_name in class_names:
        
        divided_col_by_classes.append(
            train_data.loc[train_data[label_col_name] == class_name, feature_col_name].to_frame()
        )
    
    entropy_arr = []    # this array have element, the count is same with class count.
    for val_list_by_class in divided_col_by_classes:
        
        entropy_arr.append(
            cb.Appetizer.entropy(
                cb,
                val_list_by_class
            )
        )
    return sum(entropy_arr)




# print(
#     cb.Appetizer.entropy_from_list(
#         cb,
#         divided_col_by_class[0]
#     )
# )
# print(
#     cb.Appetizer.entropy_from_list(
#         cb,
#         divided_col_by_class[1]
#     )
# )


# class ignored entropy
print('%15s' % 'Pclass : ' + str(cb.Appetizer.entropy(cb, train[['Pclass']])))              # 1.21
print('%15s' % 'Sex : ' + str(cb.Appetizer.entropy(cb, train[['Sex']])))                 # 1.08
print('%15s' % 'Age : ' + str(cb.Appetizer.entropy(cb, train[['Age']])))                 # 1.93
print('%15s' % 'Fare : ' + str(cb.Appetizer.entropy(cb, train[['Fare']])))                # 3.63
print('%15s' % 'SibSp : ' + str(cb.Appetizer.entropy(cb, train[['SibSp']])))               # 3.65
print('%15s' % 'Parch : ' + str(cb.Appetizer.entropy(cb, train[['Parch']])))               # 4.23
print('%15s' % 'Embarked : ' + str(cb.Appetizer.entropy(cb, train[['Embarked']])))            # 1.70
print('%15s' % 'Name : ' + str(cb.Appetizer.entropy(cb, train[['Name']])))                # 1.00 : it means this feature is not helpful

print("\n")
# summary class entropy indepently
print('%15s' % 'Pclass : ' + str(entropy_independently(train, label_col_name, "Pclass")))   # 2.56
print('%15s' % 'Sex : ' + str(entropy_independently(train, label_col_name, "Sex")))      # 2.62
print('%15s' % 'Age : ' + str(entropy_independently(train, label_col_name, "Age")))      # 3.32
print('%15s' % 'Fare : ' + str(entropy_independently(train, label_col_name, "Fare")))     # 5.76
print('%15s' % 'SibSp : ' + str(entropy_independently(train, label_col_name, "SibSp")))    # 6.34
print('%15s' % 'Parch : ' + str(entropy_independently(train, label_col_name, "Parch")))    # 7.27
print('%15s' % 'Embarked : ' + str(entropy_independently(train, label_col_name, "Embarked"))) # 3.36
print('%15s' % 'Name : ' + str(entropy_independently(train, label_col_name, "Name")))     # 2.00

