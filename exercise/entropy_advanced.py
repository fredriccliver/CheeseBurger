import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) +'/..'))

import lib.CheeseBurger as cb
import pandas as pd
import seaborn as sb


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
    #   Pclass : 1.2152690390625296
    #      Sex : 1.087127667748693
    #      Age : 1.9319414040125862
    #     Fare : 3.6396487634796717
    #    SibSp : 3.6573252162477754
    #    Parch : 4.233033666254767
    # Embarked : 1.7039494078988158
    #     Name : 1.0000000000000053

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
    #   Pclass : 2.566516018393548
    #      Sex : 2.62836868536768
    #      Age : 3.3288721221155564
    #     Fare : 5.767089465051383
    #    SibSp : 6.34267182323186
    #    Parch : 7.272555834612472
    # Embarked : 3.3626349582398296
    #     Name : 2.0000000000000115

# 클래스 별로 엔트로피를 따로 구해서 합산 했지만 클래스를 무시했을 때와 크게 다른 점이 없음.
# 추후, class * feature 로 entropy 를 따로 계산해서 2차원 배열로 entropy 를 구성해서 point 계산시 따로 적용하는 것을 고려해야 함.
# titanic prediction 은 two class 라서 iris prediction 처럼 multi class classification 으로 다시 entropy 를 계산해 보아야 함.

sb.barplot(x = "Pclass",y = "Survived",data=train)

