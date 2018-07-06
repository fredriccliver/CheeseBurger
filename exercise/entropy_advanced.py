import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) +'/..'))

import lib.CheeseBurger as cb
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

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

# 일단은 클래스별 합산 엔트로피를 이용하기로 하였지만, 추후 가중치 벡터를 Class * Feature 의 2 dimension matrix 로 사용해야할 수도 있음.
# entropy_independently > get_feature_weight 로 이름 변경함.
# 결과로 나온 값이 '분산도' 보다는 '가중치'로 부르는 것이 나을 것 같아서 변경함.
def get_feature_weight(train_data: list, label_col_name: str, feature_col_name: str):
    '''
    arguments
        train_data: list, 
        label_col_name: str, 
        feature_col_name: str
    '''
    class_names = list(train_data[[label_col_name]].groupby(label_col_name).size().keys())
    levels = list(train_data[[feature_col_name]].groupby(feature_col_name).size().keys())
    
    divided_col_by_classes = []
    
    for level in levels:
        
        divided_col_by_classes.append(
            # train_data.loc[train_data[label_col_name] == class_name, feature_col_name].to_frame()
            list(train_data.loc[train_data[feature_col_name] == level, [label_col_name,feature_col_name]].groupby(label_col_name).size())
        )
    
    # entropy_arr = []    # this array have element, the count is same with class count.
    # for val_list_by_class in divided_col_by_classes:
        
    #     entropy_arr.append(
    #         cb.Appetizer.entropy(
    #             cb,
    #             val_list_by_class
    #         )
    #     )
    # return sum(entropy_arr)
    summary_entropy = 0
    for val_list_by_class in divided_col_by_classes:
        summary_entropy += cb.Appetizer.entropy_from_list(
            cb,
            val_list_by_class
        )
    return summary_entropy



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


'''

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

'''

# 클래스 별로 엔트로피를 따로 구해서 합산 했지만 클래스를 무시했을 때와 크게 다른 점이 없음.
# 추후, class * feature 로 entropy 를 따로 계산해서 2차원 배열로 entropy 를 구성해서 point 계산시 따로 적용하는 것을 고려해야 함.
# titanic prediction 은 two class 라서 iris prediction 처럼 multi class classification 으로 다시 entropy 를 계산해 보아야 함.

# class를 무시하고 data 분포만을 확인한 entropy 값은 아무런 의미가 없음.



d = {
    'sex': ['man', 'man', 'man', 'man', 'man', 'women', 'women', 'women', 'women', 'women' ],
    'survived' : [0,0,0,0,1,1,1,1,1,0]
}
df_women_survived = pd.DataFrame(data=d)

d = {
    'sex': ['man', 'man', 'man', 'man', 'man', 'women', 'women', 'women', 'women', 'women' ],
    'survived' : [0,0,1,0,1,0,0,1,0,1]
}
df_equality = pd.DataFrame(data=d)

d = {
    'sex': ['man', 'man', 'man', 'man', 'man', 'women', 'women', 'women', 'women', 'women' ],
    'survived' : [0,0,0,0,0,1,1,1,1,1]
}
df_conditional_genocide = pd.DataFrame(data=d)

d = {
    'sex': ['man', 'man', 'man', 'man', 'man', 'women', 'women', 'women', 'women', 'women' ],
    'survived' : [1,0,1,0,1,0,0,1,1,1]
}
df_onlyluck = pd.DataFrame(data=d)

d = {
    'name': ['man1', 'man2', 'man3', 'man4', 'man5', 'women6', 'women7', 'women8', 'women9', 'women0' ],
    'survived' : [1,0,1,0,1,0,0,1,1,1]
}
df_superleveled = pd.DataFrame(data=d)

print(cb.Classifier.get_feature_weight(cb, df_women_survived, 'survived', 'sex'))
print(cb.Classifier.get_feature_weight(cb, df_equality, 'survived', 'sex'))
print(cb.Classifier.get_feature_weight(cb, df_conditional_genocide, 'survived', 'sex'))
print(cb.Classifier.get_feature_weight(cb, df_onlyluck, 'survived', 'sex'))
print(cb.Classifier.get_feature_weight(cb, df_superleveled, 'survived', 'name'))

print("\n")

# print(cb.Classifier.get_feature_weight(cb, df_women_survived, 'survived', 'sex'))     # 높아야 함
# print(cb.Classifier.get_feature_weight(cb, df_equality, 'survived', 'sex'))           # 가장 낮아야 함
# print(cb.Classifier.get_feature_weight(cb, df_genocide, 'survived', 'sex'))           # 가장 높아야 함.
# 1.3600000000000003
# 1.04
# 2.0

train['FamilySize'] = train['SibSp'] + train['Parch']
train['SexAndPclass'] = train['Sex'] + train['Pclass'].astype("str")
train['Survived2'] = train['Survived']
train['Generation'] = np.floor(train['Age']/10)

print('%15s' % 'Pclass : ' + str(cb.Classifier.get_feature_weight(cb, train, label_col_name, "Pclass")))
print('%15s' % 'Sex : ' + str(cb.Classifier.get_feature_weight(cb, train, label_col_name, "Sex")))
print('%15s' % 'Age : ' + str(cb.Classifier.get_feature_weight(cb, train, label_col_name, "Age")))
print('%15s' % 'Generation : ' + str(cb.Classifier.get_feature_weight(cb, train, label_col_name, "Generation")))
print('%15s' % 'FamilySize : ' + str(cb.Classifier.get_feature_weight(cb, train, label_col_name, "FamilySize")))
print('%15s' % 'SexAndPclass : ' + str(cb.Classifier.get_feature_weight(cb, train, label_col_name, "SexAndPclass")))
print('%15s' % 'Survived2 : ' + str(cb.Classifier.get_feature_weight(cb, train, label_col_name, "Survived2")))
print('%15s' % 'Fare : ' + str(cb.Classifier.get_feature_weight(cb, train, label_col_name, "Fare")))
print('%15s' % 'SibSp : ' + str(cb.Classifier.get_feature_weight(cb, train, label_col_name, "SibSp")))
print('%15s' % 'Parch : ' + str(cb.Classifier.get_feature_weight(cb, train, label_col_name, "Parch")))
print('%15s' % 'Embarked : ' + str(cb.Classifier.get_feature_weight(cb, train, label_col_name, "Embarked")))
print('%15s' % 'Name : ' + str(cb.Classifier.get_feature_weight(cb, train, label_col_name, "Name")))
#       Pclass : 0.427752669732258
#          Sex : 0.65709217134192
#          Age : 0.02344504420139277
#   Generation : 0.19954562083654162
#   FamilySize : 0.4092510253202495
# SexAndPclass : 0.27939256359780434
#    Survived2 : 1.0
#         Fare : 0.017156030920090673
#        SibSp : 0.522468614966077
#        Parch : 0.5929023464116023
#     Embarked : 0.5604391597066383
#         Name : 0.002372735116479737

print("\n")


# scalar weight 말고, feature weight vector 를 출력.
print('%15s' % 'Pclass : ' + str(cb.Classifier.get_weight(cb, train, label_col_name, "Pclass", 1)))
print('%15s' % 'Sex : ' + str(cb.Classifier.get_feature_weight_vector(cb, train, label_col_name, "Sex")))
print('%15s' % 'Age : ' + str(cb.Classifier.get_feature_weight_vector(cb, train, label_col_name, "Age")))
print('%15s' % 'Generation : ' + str(cb.Classifier.get_feature_weight_vector(cb, train, label_col_name, "Generation")))
print('%15s' % 'FamilySize : ' + str(cb.Classifier.get_feature_weight_vector(cb, train, label_col_name, "FamilySize")))
print('%15s' % 'SexAndPclass : ' + str(cb.Classifier.get_feature_weight_vector(cb, train, label_col_name, "SexAndPclass")))
print('%15s' % 'Survived2 : ' + str(cb.Classifier.get_feature_weight_vector(cb, train, label_col_name, "Survived2")))
print('%15s' % 'Fare : ' + str(cb.Classifier.get_feature_weight_vector(cb, train, label_col_name, "Fare")))
print('%15s' % 'SibSp : ' + str(cb.Classifier.get_feature_weight_vector(cb, train, label_col_name, "SibSp")))
print('%15s' % 'Parch : ' + str(cb.Classifier.get_feature_weight_vector(cb, train, label_col_name, "Parch")))
print('%15s' % 'Embarked : ' + str(cb.Classifier.get_feature_weight_vector(cb, train, label_col_name, "Embarked")))
print('%15s' % 'Name : ' + str(cb.Classifier.get_feature_weight_vector(cb, train, label_col_name, "Name")))
#       Pclass : [0.5115875527951135, 0.34391778666940254]
#          Sex : [0.748454716474066, 0.565729626209774]
#          Age : [0.0234180313278747, 0.02347205707491084]
#   Generation : [0.20912691349234605, 0.1899643281807372]
#   FamilySize : [0.48980262175639777, 0.3286994288841012]
# SexAndPclass : [0.3631009850664066, 0.19568414212920213]
#    Survived2 : [1.0, 1.0]
#         Fare : [0.021556000145984848, 0.0127560616941965]
#        SibSp : [0.5589928367855448, 0.48594439314660925]
#        Parch : [0.6717661852482242, 0.5140385075749805]
#     Embarked : [0.6309302225274634, 0.4899480968858131]
#         Name : [0.001821493624772325, 0.0029239766081871487]

# 현재 가중치 구하는 식이 같은 분포여도 data 수에 따라 영향받는 등 문제가 있지만,
# 일단 burger matrix 부터 만들고 prediction 해보는 걸로.

